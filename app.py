import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas as pd
import json
import io
import os
import requests
import re
import tempfile
import time

# ReportLab (PDF 생성)
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# 1. 페이지 설정
st.set_page_config(page_title="학생부 AI 전문가 심층 분석", layout="wide")

# --- [Session State] ---
if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'df_activities' not in st.session_state: st.session_state['df_activities'] = None
if 'rec_data' not in st.session_state: st.session_state['rec_data'] = None
if 'uploader_key' not in st.session_state: st.session_state['uploader_key'] = 0

# --- [안전 필터 설정 (핵심 수정)] ---
# 학생부 내의 '마약(책제목)', '질병', '독성' 등의 단어로 인해 AI가 답변을 차단하는 것을 방지합니다.
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- [기능 함수들] ---
def find_best_model_dynamically(api_key):
    genai.configure(api_key=api_key)
    try:
        all_models = genai.list_models()
        valid_models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        
        if not valid_models: raise Exception("사용 가능한 모델이 없습니다.")

        flash_models = [m for m in valid_models if 'gemini-1.5-flash' in m]
        if flash_models: return flash_models[0]

        pro_models = [m for m in valid_models if 'gemini-1.5-pro' in m]
        if pro_models: return pro_models[0]

        gemini_models = [m for m in valid_models if 'gemini' in m]
        if gemini_models: return gemini_models[0]
            
        return valid_models[0]
    except Exception as e:
        return "models/gemini-1.5-flash"

def reset_analysis():
    st.session_state['analyzed'] = False
    st.session_state['df_activities'] = None
    st.session_state['rec_data'] = None
    st.session_state['uploader_key'] += 1

def extract_text_from_excel(file):
    try:
        df = pd.read_excel(file, header=None)
        text = ""
        for i, row in df.iterrows():
            r_text = " ".join([str(v) for v in row if pd.notna(v) and str(v).strip()!=""])
            if r_text: text += r_text + "\n"
        return text
    except: return ""

def refine_dataframe(df):
    rename = {'요약':'내용', '활동내용':'내용', 'summary':'내용', '상세내용':'내용'}
    df = df.rename(columns=rename)
    req_cols = ['학년', '영역', '세부항목', '내용']
    for c in req_cols:
        if c not in df.columns: df[c] = "-"
    
    for i, row in df.iterrows():
        content = str(row['내용'])
        if str(row['학년']) in ["-", "", "nan"]:
            m = re.search(r'([1-3])학년', content)
            if m: df.at[i, '학년'] = f"{m.group(1)}학년"
        if str(row['세부항목']) in ["-", "", "nan"]:
            m_b = re.search(r'^\[(.*?)\]', content)
            if m_b:
                df.at[i, '세부항목'] = m_b.group(1)
                df.at[i, '내용'] = content.replace(f"[{m_b.group(1)}]", "").strip()
    return df[req_cols]

def parse_json_safely(response):
    """AI 응답을 안전하게 JSON으로 파싱합니다."""
    try:
        text = response.text
    except ValueError:
        # Safety 필터에 걸렸을 때 발생하는 에러 처리
        raise Exception("AI가 안전 정책(Safety) 문제로 응답을 거부했습니다. (학생부 내 민감 단어 포함 의심)")
        
    try:
        text = text.strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        raise Exception(f"JSON 변환 실패. AI가 잘못된 형식을 반환했습니다.\n원본 일부: {text[:200]}")

# --- [PDF 생성 함수] ---
def register_korean_font():
    font_name = "NanumGothic"
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        try:
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            resp = requests.get(url)
            with open(font_path, "wb") as f: f.write(resp.content)
        except: pass
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
        return font_name
    except: return "Helvetica"

def create_output_pdf(df, rec_data):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=10*mm, leftMargin=10*mm, topMargin=15*mm, bottomMargin=15*mm)
    font_name = register_korean_font()
    styles = getSampleStyleSheet()
    
    s_title = ParagraphStyle('T', parent=styles['Heading1'], fontName=font_name, fontSize=18, alignment=1, spaceAfter=20)
    s_head = ParagraphStyle('H', parent=styles['Heading2'], fontName=font_name, fontSize=14, spaceBefore=15, spaceAfter=10)
    s_body = ParagraphStyle('B', parent=styles['Normal'], fontName=font_name, fontSize=10, leading=15)
    s_cell_c = ParagraphStyle('CC', parent=styles['Normal'], fontName=font_name, fontSize=9, alignment=1)
    s_cell_l = ParagraphStyle('CL', parent=styles['Normal'], fontName=font_name, fontSize=9, alignment=0)

    elems = []
    elems.append(Paragraph("학생부 전문가 심층 분석 보고서", s_title))
    
    elems.append(Paragraph("1. 종합 분석 및 학과 추천", s_head))
    elems.append(Paragraph(rec_data.get('종합분석', ''), s_body))
    elems.append(Spacer(1, 10))
    for idx, r in enumerate(rec_data.get('추천학과', []), 1):
        elems.append(Paragraph(f"<b>{idx}. {r.get('학과명')}</b>", s_body))
        elems.append(Paragraph(f" - {r.get('추천이유')}", s_body))
        elems.append(Spacer(1, 5))
    elems.append(PageBreak())

    elems.append(Paragraph("2. 영역별 핵심 활동 팩트 (3단계 검증 완료)", s_head))
    data = [[Paragraph('<b>학년</b>', s_cell_c), Paragraph('<b>영역</b>', s_cell_c), Paragraph('<b>세부항목</b>', s_cell_c), Paragraph('<b>내용</b>', s_cell_c)]]
    if df is not None:
        for _, row in df.iterrows():
            data.append([
                Paragraph(str(row['학년']), s_cell_c),
                Paragraph(str(row['영역']), s_cell_c),
                Paragraph(str(row['세부항목']), s_cell_c),
                Paragraph(str(row['내용']), s_cell_l)
            ])
    
    table = Table(data, colWidths=[12*mm, 28*mm, 35*mm, 115*mm], repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.navy),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elems.append(table)
    doc.build(elems)
    buffer.seek(0)
    return buffer

# --- [UI 구성] ---
st.title("🎓 학생부 AI 전문가 심층 분석")
st.markdown("RAG 기반 팩트 체크 및 AI 전문가 교차 검증 시스템 (안전 필터 해제 적용)")

with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Google API Key", type="password")
    st.divider()
    st.button("🔄 초기화", on_click=reset_analysis)

uploaded_file = st.file_uploader("학생부 파일 (PDF/Excel/TXT)", type=["pdf","txt","xlsx","xls"], key=f"up_{st.session_state.uploader_key}")

# --- [메인 로직] ---
if uploaded_file and api_key:
    if not st.session_state.analyzed:
        if st.button("🚀 전문가 3인 분석 시작"):
            
            is_pdf = uploaded_file.name.lower().endswith('.pdf')
            
            with st.status("🕵️ 분석 프로세스 진행 중...", expanded=True) as status:
                try:
                    found_model_name = find_best_model_dynamically(api_key)
                    model = genai.GenerativeModel(found_model_name)
                    st.write("⚙️ 최적 모델 검색 완료")
                    
                    analysis_input = []
                    
                    if is_pdf:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
                            tf.write(uploaded_file.getvalue())
                            tpath = tf.name
                        
                        gfile = genai.upload_file(tpath, mime_type="application/pdf")
                        
                        while gfile.state.name == "PROCESSING": 
                            time.sleep(1)
                            gfile = genai.get_file(gfile.name)
                            
                        analysis_input = [gfile]
                        os.unlink(tpath)
                        st.write("📂 PDF 스캔 및 멀티모달 인식 완료")
                    else:
                        ext = uploaded_file.name.split('.')[-1].lower()
                        txt = extract_text_from_excel(uploaded_file) if ext in ['xlsx','xls'] else io.StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                        if not txt.strip(): raise Exception("빈 파일입니다.")
                        analysis_input = [txt]
                        st.write("📂 텍스트 추출 완료")

                    # --- [Step 1: 1차 추출] ---
                    st.write("📝 Step 1: 기초 데이터 추출 (Analyst Agent)")
                    prompt_draft = """
                    당신은 생기부 분석가입니다. [자료]에서 아래 항목만 추출하세요.
                    - 대상: 자율활동, 동아리활동, 진로활동, 세특, 행발 (수상/독서 제외)
                    - 형식: `[세부활동명] 내용`
                    - 필수키: 학년, 영역, 세부항목, 내용
                    """
                    # JSON 강제 모드 적용
                    config_json = genai.GenerationConfig(response_mime_type="application/json")
                    
                    resp_draft = model.generate_content(
                        [prompt_draft] + analysis_input, 
                        generation_config=config_json,
                        safety_settings=SAFETY_SETTINGS
                    )
                    draft_data = parse_json_safely(resp_draft)
                    
                    # --- [Step 2: 3인 전문가 교차 검증] ---
                    st.write("🔍 Step 2: 전문가 교차 검증 (Verification Committee)")
                    prompt_verify = f"""
                    당신은 3명의 생기부 검증 위원회입니다. [기초 데이터]를 [원본]과 대조해 검증하세요.

                    [역할 분담]
                    1. 팩트체커: 원본에 없는 내용(거짓) 삭제.
                    2. 에디터: `[활동명] 핵심 성과 + 내용`으로 1~2문장 요약. (명사형 종결)
                    3. 입학사정관: 수상/독서 내역 삭제. 학년/영역 오분류 수정.

                    [기초 데이터]
                    {json.dumps(draft_data, ensure_ascii=False)}
                    """
                    resp_verify = model.generate_content(
                        [prompt_verify] + analysis_input, 
                        generation_config=config_json,
                        safety_settings=SAFETY_SETTINGS
                    )
                    verified_data = parse_json_safely(resp_verify)
                    
                    df = refine_dataframe(pd.DataFrame(verified_data))
                    st.write("✅ 검증 완료: 팩트 체크 및 문장 정제 끝")

                    # --- [Step 3: 최종 종합 분석] ---
                    st.write("🎓 Step 3: 진로 적합성 종합 분석 (Career Advisor)")
                    prompt_final = f"""
                    데이터: {json.dumps(verified_data, ensure_ascii=False)}
                    위 데이터를 바탕으로:
                    1. 추천 학과 3개 (활동 근거 포함)
                    2. 종합 분석 의견
                    
                    반드시 아래 JSON 구조로만 응답하세요.
                    {{
                        "종합분석": "...",
                        "추천학과": [
                            {{"학과명": "...", "추천이유": "..."}}
                        ]
                    }}
                    """
                    resp_final = model.generate_content(
                        prompt_final, 
                        generation_config=config_json,
                        safety_settings=SAFETY_SETTINGS
                    )
                    rec_data = parse_json_safely(resp_final)

                    st.session_state['df_activities'] = df
                    st.session_state['rec_data'] = rec_data
                    st.session_state['analyzed'] = True
                    
                    status.update(label="🎉 모든 분석이 완료되었습니다!", state="complete", expanded=False)

                except Exception as e:
                    st.error(f"에러 발생: {e}")
                    status.update(label="⚠️ 오류 발생", state="error")

    # --- [결과 화면] ---
    if st.session_state.analyzed:
        df = st.session_state['df_activities']
        rec = st.session_state['rec_data']

        tab1, tab2 = st.tabs(["📄 검증된 활동 팩트", "🎓 AI 입학사정관 리포트"])
        
        with tab1:
            st.info("✅ 3단계 검증(팩트체크/문장정제/분류검수)을 거친 데이터입니다.")
            st.dataframe(df, use_container_width=True, height=500)
        
        with tab2:
            st.success("🎯 **종합 분석 의견**")
            st.write(rec.get('종합분석'))
            st.divider()
            cols = st.columns(3)
            for i, r in enumerate(rec.get('추천학과', [])):
                with cols[i]:
                    st.markdown(f"### 🏫 {i+1}. {r.get('학과명', '추천학과')}")
                    st.caption(r.get('추천이유', ''))

        st.divider()
        c1, c2 = st.columns(2)
        excel_bio = io.BytesIO()
        with pd.ExcelWriter(excel_bio) as w:
            df.to_excel(w, index=False, sheet_name='활동팩트')
            pd.DataFrame(rec.get('추천학과', [])).to_excel(w, index=False, sheet_name='학과추천')
        
        c1.download_button("📊 엑셀 다운로드", excel_bio.getvalue(), "분석결과.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        
        try:
            pdf_bio = create_output_pdf(df, rec)
            c2.download_button("📄 PDF 보고서 다운로드", pdf_bio, "분석보고서.pdf", "application/pdf", use_container_width=True)
        except: c2.error("PDF 생성 실패")

elif not uploaded_file:
    st.info("파일을 업로드하면 분석 버튼이 나타납니다.")