import os

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

# --- LLMに問い合わせる関数 ---
# 「入力テキスト」と「ラジオボタンでの選択値」を引数に取り、
# LLM からの回答テキストを戻り値として返します。
def ask_expert(user_text: str, expert_type: str) -> str:
    # 選択された専門家ごとにシステムメッセージ（役割）を変更
    if expert_type == "営業":
        system_prompt = (
            "あなたは一流の営業コンサルタントです。"
            "法人営業・個人営業どちらにも精通しており、"
            "具体的なトーク例や行動レベルのアドバイスをわかりやすく提案してください。"
        )
    elif expert_type == "マーケティング":
        system_prompt = (
            "あなたは一流のマーケティング専門家です。"
            "デジタル広告、SNS運用、コンテンツマーケティング、"
            "LTV向上などを踏まえた具体的な施策を提案してください。"
        )
    else:
        # 想定外の値が来た場合の保険
        system_prompt = "あなたはビジネス全般に詳しいコンサルタントです。"

    # LangChain の LLM インスタンスを作成
    # OPENAI_API_KEY は環境変数または Streamlit の secrets に設定しておく
    api_key = os.getenv("OPENAI_API_KEY", None)
    if api_key is None and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]

    result = llm.invoke(messages)
    return result.content


# --- Streamlit で画面を定義 ---
st.set_page_config(
    page_title="営業・マーケティング専門家に相談できるアプリ",
    page_icon="💬",
)

st.title("💬 専門家に相談できるアプリ")

st.write(
    """
このアプリでは、**営業** または **マーケティング** の専門家に
ビジネスの悩みを相談できます。

- ラジオボタンで相談したい専門家の種類を選択
- テキストボックスに相談内容を入力
- 「相談する」ボタンを押すと、LLM が専門家として回答します
"""
)

st.markdown("---")

# 入力フォーム（1つのフォームでまとめて送信）
with st.form("expert_form"):
    expert_type = st.radio(
        "相談する専門家を選んでください",
        ("営業", "マーケティング"),
        horizontal=True,
    )

    user_text = st.text_area(
        "相談内容（できるだけ具体的に書いてください）",
        height=160,
        placeholder=(
            "例：\n"
            "・新規開拓のクロージング率を上げるための改善案が知りたい\n"
            "・BtoB SaaS のリード獲得のために、どんなマーケ施策を優先すべきか など"
        ),
    )

    submitted = st.form_submit_button("専門家に相談する")

# ボタンが押されたときの処理
if submitted:
    if not user_text.strip():
        st.warning("相談内容を入力してください。")
    else:
        with st.spinner("専門家が回答を考えています…"):
            answer = ask_expert(user_text, expert_type)

        st.markdown("### ✅ 専門家からの回答")
        st.write(answer)

st.markdown("---")

st.markdown(
    """
### ℹ️ このWebアプリについて

- 営業またはマーケティングの専門家ロールを、LLM（gpt-4o-mini）に割り当てています  
- LangChain を使って、システムメッセージとユーザー入力を LLM に渡しています  
- 画面上のフォームからテキストを送信すると、回答が下部に表示されます  

Streamlit Community Cloud にデプロイする際は、Python のバージョンを **3.11** に設定し、
`OPENAI_API_KEY` をシークレットに登録してご利用ください。
"""
)