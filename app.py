import streamlit as st
import io, json
import numpy as np
from PIL import Image, ImageOps
import google.generativeai as genai
from rembg import remove
from supabase import create_client

st.set_page_config(page_title="Wardrobe.", page_icon="✦", layout="wide")

# Apple 級極簡風格 CSS 注入
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, sans-serif;
    }
    
    .stApp {
        background-color: #F5F5F7;
    }

    /* 卡片容器 */
    div[data-testid="stColumn"] > div {
        background: #FFFFFF;
        padding: 16px;
        border-radius: 18px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        border: 1px solid #E5E5EA;
        transition: all 0.25s ease;
    }
    
    div[data-testid="stColumn"] > div:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }

    /* 強制圖片容器統一為正方形與等高 */
    div[data-testid="stImage"] img {
        border-radius: 12px;
        object-fit: contain;
        aspect-ratio: 1 / 1;
        width: 100%;
        background-color: #FAFAFC;
    }

    /* Apple 風格全寬按鈕修復 */
    .stButton>button {
        border-radius: 12px;
        border: 1px solid #D1D1D6;
        background-color: #FFFFFF;
        color: #1D1D1F;
        font-weight: 500;
        width: 100%;
        white-space: nowrap;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0071E3;
        color: #FFFFFF;
        border-color: #0071E3;
    }
</style>
''', unsafe_allow_html=True)

# 💧 清洗 Supabase URL
raw_url = str(st.secrets.get("SUPABASE_URL", ""))
clean_url = raw_url.replace("/rest/v1", "").replace("/rest", "").strip().rstrip("/")
supabase_key = str(st.secrets.get("SUPABASE_KEY", "")).strip()
gemini_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()

def get_supabase():
    return create_client(clean_url, supabase_key)

if gemini_key:
    genai.configure(api_key=gemini_key)

@st.cache_data(ttl=3600)
def get_available_gemini_models():
    models_to_try = []
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models_to_try.append(m.name)
    except Exception:
        pass
    
    default_candidates = [
        'models/gemini-1.5-flash',
        'models/gemini-2.0-flash',
        'models/gemini-1.5-flash-latest',
        'models/gemini-1.5-pro'
    ]
    for c in default_candidates:
        if c not in models_to_try:
            models_to_try.append(c)
            
    return models_to_try

DETAILED_CATEGORIES = [
    "T恤/背心", "襯衫/雪紡", "針織/毛衣", "帽T/衛衣",
    "牛仔褲", "休閒長褲", "短褲/五分褲", "裙子",
    "洋裝/連身褲", "外套/夾克", "鞋款", "包包/配件"
]

STATUS_OPTIONS = ["正常", "預計上架", "已上架", "拜拜"]

# ----------------- AI 視覺辨識 + 旋轉角度判斷 -----------------
def analyze_garment_creative(pil_img):
    if not gemini_key:
        return None, "⚠️ 請在 Streamlit Secrets 設定 GEMINI_API_KEY"

    prompt = f'''你是一位說話風趣直白、帶點幽默無厘頭的潮流時尚觀察家。請觀察這張衣服照片：

1. 【姿勢與旋轉判斷】：請觀察這件衣服的領口/頂部朝向哪裡。為了讓衣服變成「領口朝上、垂直擺正」的正方向，請告訴我需要【順時針旋轉多少度】？
   可選數值只有四種：0, 90, 180, 270。

2. 【創意幽默名稱】：為這件單品取一個【幽默、無厘頭、微酸或搞怪】的繁體中文名稱！
   ⚠️【字數嚴格限制】：名稱「絕對不能超過 13 個字」！（上限 13 字）。

3. 【精準分類】：請「嚴格」從以下選單中選擇最正確的類別：
{json.dumps(DETAILED_CATEGORIES, ensure_ascii=False)}

請輸出純 JSON 格式：
{{
  "rotate_angle": 0,
  "name": "創意幽默名稱(最多13字)",
  "category": "必須完全吻合分類選單",
  "color": "主要顏色",
  "season": "四季/夏季/春秋/冬季",
  "style": "休閒/街頭/正式/極簡"
}}'''

    available_models = get_available_gemini_models()
    last_err = ""

    for model_name in available_models:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content([pil_img, prompt])
            if resp and resp.text:
                cleaned = resp.text.replace("```json", "").replace("```", "").strip()
                meta = json.loads(cleaned)
                
                if "name" in meta and len(meta["name"]) > 13:
                    meta["name"] = meta["name"][:13]
                    
                return meta, None
        except Exception as e:
            last_err = str(e)
            continue

    return None, f"AI 連線失敗: {last_err}"

# ----------------- 🎯 高清流暢去背處理 -----------------
def clean_bg_removal(pil_img, canvas_size=1000):
    # 使用 rembg 進行高品質去背
    rgba = remove(pil_img)
    
    # 自動偵測有效圖像範圍（自動裁切掉多餘空白）
    bbox = rgba.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)

    # 建立 1000x1000 正方形畫布並精緻置中
    square_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    target_size = canvas_size - 120  # 留白邊距
    
    w, h = rgba.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    
    resized_item = rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2
    
    square_canvas.paste(resized_item, (offset_x, offset_y), resized_item)

    buf = io.BytesIO()
    square_canvas.save(buf, format="PNG")
    return buf.getvalue()

# 旋轉圖片
def rotate_image_bytes(image_url, angle):
    import requests
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content)).convert("RGBA")
    rotated_img = img.rotate(-angle, expand=True)
    
    canvas_size = 1000
    square_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    target_size = canvas_size - 120
    
    bbox = rotated_img.getbbox()
    if bbox:
        rotated_img = rotated_img.crop(bbox)

    w, h = rotated_img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized_item = rotated_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2
    square_canvas.paste(resized_item, (offset_x, offset_y), resized_item)
    
    buf = io.BytesIO()
    square_canvas.save(buf, format="PNG")
    return buf.getvalue()

# ----------------- 資料庫存取 -----------------
def fetch_clothes():
    try:
        sp = get_supabase()
        res = sp.table("clothes").select("*").order("created_at", desc=True).execute()
        return res.data or [], None
    except Exception as e:
        return [], str(e)

st.title("Wardrobe.")
st.caption("Apple Style Minimalist Closet & Resale Manager")

tab1, tab2, tab3, tab4 = st.tabs(["👗 典藏與衣櫥", "💰 二手拍賣管理", "📸 快速上傳", "💬 AI 時尚問答"])

# ==================== Tab 1: 典藏衣櫥 ====================
with tab1:
    clothes, err = fetch_clothes()
    if err:
        st.error("連線失敗：" + str(err))
    elif not clothes:
        st.info("💡 目前衣櫥空空如也，快點擊「📸 快速上傳」加入新單品！")
    else:
        valid_status_default = ["正常", "預計上架", "已上架"]
        
        c_cat, c_status = st.columns(2)
        with c_cat:
            selected_cat = st.selectbox("🏷️ 品類篩選", ["全部"] + DETAILED_CATEGORIES, key="tab1_cat")
        with c_status:
            selected_status_filter = st.multiselect("📌 狀態篩選", STATUS_OPTIONS, default=valid_status_default, key="tab1_status")

        filtered_items = clothes
        if selected_status_filter:
            filtered_items = [item for item in filtered_items if item.get("status", "正常") in selected_status_filter]
        else:
            filtered_items = []

        if selected_cat != "全部":
            filtered_items = [item for item in filtered_items if selected_cat in item.get("category", "")]

        st.caption(f"共展示 {len(filtered_items)} 件單品")
        st.divider()

        cols_per_row = 4
        for row_idx in range(0, len(filtered_items), cols_per_row):
            row_items = filtered_items[row_idx:row_idx+cols_per_row]
            cols = st.columns(cols_per_row)
            for col, item in zip(cols, row_items):
                with col:
                    img_url = item.get('image_url', '')
                    if img_url:
                        st.image(img_url, use_container_width=True)
                    st.markdown(f"**{item.get('name', '未命名')}**")
                    
                    status_curr = item.get('status', '正常')
                    badge_color = "🟢" if status_curr == "正常" else ("🟡" if status_curr == "預計上架" else ("🔵" if status_curr == "已上架" else "🔴"))
                    st.caption(f"🏷️ {item.get('category','未分類')} · {badge_color} {status_curr}")
                    
                    with st.expander("⚙️ 編輯 / 管理"):
                        new_name = st.text_input("品名 (最多13字)", item.get('name',''), key=f"n_{item['id']}")
                        current_cat = item.get('category', '休閒長褲')
                        cat_idx = DETAILED_CATEGORIES.index(current_cat) if current_cat in DETAILED_CATEGORIES else 0
                        new_cat = st.selectbox("分類", DETAILED_CATEGORIES, index=cat_idx, key=f"c_{item['id']}")
                        
                        st_idx = STATUS_OPTIONS.index(status_curr) if status_curr in STATUS_OPTIONS else 0
                        new_status = st.selectbox("狀態", STATUS_OPTIONS, index=st_idx, key=f"st_{item['id']}")
                        
                        orig_p = st.number_input("原價 ($)", value=int(item.get('original_price') or 0), step=100, key=f"op_{item['id']}")
                        sale_p = st.number_input("預計售價 ($)", value=int(item.get('sale_price') or 0), step=100, key=f"sp_{item['id']}")

                        # 🔄 照片手動旋轉校正按鈕
                        st.markdown("**🔄 照片方向旋轉校正：**")
                        r_col1, r_col2 = st.columns(2)
                        if r_col1.button("↺ 逆時針 90°", key=f"rot_l_{item['id']}"):
                            with st.spinner("正在旋轉圖片..."):
                                new_bytes = rotate_image_bytes(img_url, -90)
                                fname = img_url.split('/')[-1].split('?')[0]
                                get_supabase().storage.from_("clothes").upload(path=fname, file=new_bytes, file_options={"x-upsert": "true", "content-type": "image/png"})
                                st.success("已旋轉！")
                                st.rerun()
                        if r_col2.button("↻ 順時針 90°", key=f"rot_r_{item['id']}"):
                            with st.spinner("正在旋轉圖片..."):
                                new_bytes = rotate_image_bytes(img_url, 90)
                                fname = img_url.split('/')[-1].split('?')[0]
                                get_supabase().storage.from_("clothes").upload(path=fname, file=new_bytes, file_options={"x-upsert": "true", "content-type": "image/png"})
                                st.success("已旋轉！")
                                st.rerun()

                        st.markdown("---")
                        if st.button("💾 儲存修改", key=f"s_{item['id']}"):
                            try:
                                get_supabase().table("clothes").update({
                                    "name": new_name[:13],
                                    "category": new_cat,
                                    "status": new_status,
                                    "original_price": orig_p,
                                    "sale_price": sale_p
                                }).eq("id", item['id']).execute()
                                st.success("已更新！")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"儲存失敗，請確保 Supabase 已建立欄位！錯誤: {ex}")

                        if st.button("🗑️ 刪除單品", key=f"d_{item['id']}"):
                            try:
                                get_supabase().table("clothes").delete().eq("id", item['id']).execute()
                                st.warning("已刪除！")
                                st.rerun()
                            except Exception as ex:
                                st.error(f"刪除失敗: {ex}")

# ==================== Tab 2: 二手拍賣管理 ====================
with tab2:
    st.subheader("🛍️ 二手拍賣管理表")
    st.caption("此頁面僅列出【預計上架】、【已上架】及【拜拜】三種狀態的單品。")

    clothes, err = fetch_clothes()
    if err:
        st.error("連線失敗：" + str(err))
    else:
        resale_targets = ["預計上架", "已上架", "拜拜"]
        resale_items = [item for item in clothes if item.get("status", "正常") in resale_targets]

        if not resale_items:
            st.info("💡 目前沒有二手拍賣或淘汰的單品。可以在「典藏與衣櫥」將單品狀態改為「預計上架」！")
        else:
            th1, th2, th3, th4, th5 = st.columns([1, 2, 1, 1, 1.5])
            with th1: st.markdown("**小圖**")
            with th2: st.markdown("**名稱**")
            with th3: st.markdown("**原價**")
            with th4: st.markdown("**售價**")
            with th5: st.markdown("**狀態與操作**")
            st.divider()

            for item in resale_items:
                r1, r2, r3, r4, r5 = st.columns([1, 2, 1, 1, 1.5])
                with r1:
                    img_url = item.get('image_url', '')
                    if img_url:
                        st.image(img_url, width=60)
                with r2:
                    st.markdown(f"**{item.get('name', '未命名')}**")
                    st.caption(f"🏷️ {item.get('category', '未分類')}")
                with r3:
                    orig = item.get('original_price')
                    st.write(f"${orig:,}" if orig else "未填")
                with r4:
                    sale = item.get('sale_price')
                    st.write(f"**${sale:,}**" if sale else "未填")
                with r5:
                    curr_st = item.get('status', '預計上架')
                    st_idx = STATUS_OPTIONS.index(curr_st) if curr_st in STATUS_OPTIONS else 1
                    new_st = st.selectbox("", STATUS_OPTIONS, index=st_idx, key=f"resale_st_{item['id']}")
                    if new_st != curr_st:
                        try:
                            get_supabase().table("clothes").update({"status": new_st}).eq("id", item['id']).execute()
                            st.success("狀態已更新！")
                            st.rerun()
                        except Exception as ex:
                            st.error(f"更新失敗: {ex}")
                st.markdown("---")

# ==================== Tab 3: 快速上傳 ====================
with tab3:
    uploaded_files = st.file_uploader("上傳照片（自動轉正去背）", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        init_status = st.selectbox("預設初始狀態", STATUS_OPTIONS, index=0)
    with col_u2:
        init_orig_price = st.number_input("預設原價 ($)", value=0, step=100)

    if uploaded_files and st.button("🚀 開始 Apple 風格智慧歸檔"):
        sp = get_supabase()
        
        for index, file in enumerate(uploaded_files):
            with st.spinner(f"⚡ 正在進行高清精準去背 `{file.name}`..."):
                try:
                    raw_bytes = file.read()
                    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    
                    # 1. EXIF 校正
                    pil_img = ImageOps.exif_transpose(pil_img)
                    
                    # 2. AI 智慧分析與轉向判斷
                    meta, ai_err = analyze_garment_creative(pil_img)
                    if ai_err:
                        st.error(f"⚠️ {ai_err}")
                        meta = {
                            "rotate_angle": 0,
                            "name": file.name.split('.')[0][:13],
                            "category": "未分類",
                            "color": "未指定",
                            "season": "四季",
                            "style": "休閒"
                        }

                    # 3. 旋轉轉正圖片
                    rot_angle = int(meta.get("rotate_angle", 0))
                    if rot_angle in [90, 180, 270]:
                        pil_img = pil_img.rotate(-rot_angle, expand=True)

                    # 4. 🎯 高清去背
                    final_bytes = clean_bg_removal(pil_img)

                    # 5. 上傳至 Supabase Storage
                    import time
                    fname = f"item_{int(time.time())}_{index}.png"
                    sp.storage.from_("clothes").upload(path=fname, file=final_bytes, file_options={"x-upsert": "true", "content-type": "image/png"})
                    public_url = sp.storage.from_("clothes").get_public_url(fname)

                    # 6. 寫入資料庫
                    item_name = meta.get("name", "搞怪單品")[:13]
                    sp.table("clothes").insert({
                        "name": item_name,
                        "category": meta.get("category", "未分類"),
                        "color": meta.get("color", "百搭"),
                        "season": meta.get("season", "四季"),
                        "style": meta.get("style", "休閒"),
                        "image_url": public_url,
                        "status": init_status,
                        "original_price": init_orig_price,
                        "sale_price": 0
                    }).execute()

                    # 7. 🎉 展示結果
                    st.success(f"🎉【歸檔成功】")
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.image(final_bytes, width=200)
                    with res_col2:
                        st.subheader(f"✨ {item_name}")
                        st.write(f"🏷️ **精細品類**：{meta.get('category')}")
                        st.write(f"📌 **初始狀態**：{init_status}")
                        st.write(f"💰 **原價**：${init_orig_price}")
                    st.divider()

                except Exception as e:
                    st.error("❌ 處理失敗：" + str(e))

# ==================== Tab 4: AI 時尚問答 ====================
with tab4:
    st.subheader("💬 AI 穿搭顧問")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if q := st.chat_input("想問什麼穿搭問題？（如：高雄晚上去咖啡廳怎麼搭？）"):
        st.session_state.messages.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            items, _ = fetch_clothes()
            valid_items = [i for i in items if i.get("status") != "拜拜"]
            context = json.dumps(valid_items, ensure_ascii=False)
            prompt = f'''使用者目前的衣櫥單品數據：{context}
使用者提問：{q}

【重點要求】：
1. 請作為一位專業、貼心且帶點幽默感的時尚造型師給予穿搭建議。
2. ⚠️ 必須「全程使用台灣繁體中文」回答！絕對不能出現英文！
3. 請直接推薦衣櫥裡的單品名稱（如：單品名稱）並說明搭配理由。'''
            
            available_models = get_available_gemini_models()
            reply = "⚠️ AI 服務暫時無法連線，請稍後再試。"
            
            for m in available_models:
                try:
                    model = genai.GenerativeModel(m)
                    resp = model.generate_content(prompt)
                    if resp and resp.text:
                        reply = resp.text
                        break
                except Exception:
                    continue

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
