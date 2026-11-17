import hashlib
import io
import os

import requests
import streamlit as st
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Image analysis",
    page_icon=":material/image_search:",
    layout="wide",
)

# Local development: read .env. On Azure, VISION_KEY / VISION_ENDPOINT come from
# the Web App's App Settings (exposed as environment variables).
load_dotenv()

FEATURES = {
    "Caption": (VisualFeatures.CAPTION, ":material/subtitles:"),
    "Tags": (VisualFeatures.TAGS, ":material/sell:"),
    "Objects": (VisualFeatures.OBJECTS, ":material/category:"),
    "People": (VisualFeatures.PEOPLE, ":material/group:"),
    "Read (OCR)": (VisualFeatures.READ, ":material/text_fields:"),
}
DEFAULT_URL = (
    "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-sample-data-files/"
    "master/ComputerVision/Images/faces.jpg"
)
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # Image Analysis 4.0 limit is 20 MB


@st.cache_resource
def get_client() -> ImageAnalysisClient | None:
    key = os.getenv("VISION_KEY")
    endpoint = os.getenv("VISION_ENDPOINT")
    if not key or not endpoint:
        return None
    return ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))


@st.cache_data(ttl=600, max_entries=20, show_spinner=False)
def download_image(url: str) -> bytes:
    with requests.get(url, timeout=15, stream=True) as response:
        response.raise_for_status()
        data = response.raw.read(MAX_DOWNLOAD_BYTES + 1, decode_content=True)
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Image is larger than 20 MB.")
    return data


def draw_boxes(image_bytes: bytes, boxes: list[tuple[object, str]]) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, image.width // 250)
    for box, label in boxes:
        x, y, w, h = box.x, box.y, box.width, box.height
        draw.rectangle([x, y, x + w, y + h], outline="#ff4b4b", width=line_width)
        if label:
            draw.text((x + line_width + 2, y + line_width + 2), label, fill="#ff4b4b")
    return image


def show_results(result, image_bytes: bytes) -> None:
    tags = result.tags.list if result.tags is not None else None
    objects = (
        [o for o in result.objects.list if o.tags] if result.objects is not None else None
    )
    people = (
        [p for p in result.people.list if p.confidence >= 0.5]
        if result.people is not None
        else None
    )
    lines = (
        [line.text for block in result.read.blocks for line in block.lines]
        if result.read is not None
        else None
    )

    if result.caption is not None:
        st.subheader(result.caption.text.capitalize())
        st.caption(f"Confidence {result.caption.confidence:.0%}")

    sections = [
        (name, items)
        for name, items in (
            ("Tags", tags),
            ("Objects", objects),
            ("People", people),
            ("Read (OCR)", lines),
        )
        if items is not None
    ]
    if not sections:
        return

    for col, (name, items) in zip(st.columns(len(sections)), sections):
        col.metric("Text lines" if name == "Read (OCR)" else name, len(items), border=True)

    for tab, (name, _) in zip(
        st.tabs([f"{FEATURES[n][1]} {n}" for n, _ in sections]), sections
    ):
        with tab:
            if name == "Tags":
                if tags:
                    st.dataframe(
                        [{"Tag": t.name, "Confidence": t.confidence} for t in tags],
                        column_config={
                            "Confidence": st.column_config.ProgressColumn(
                                "Confidence", min_value=0, max_value=1, format="percent"
                            )
                        },
                        hide_index=True,
                        alt="Tags detected with their confidence",
                    )
                else:
                    st.caption("No tags found.")
            elif name == "Objects":
                if objects:
                    boxes = [
                        (o.bounding_box, f"{o.tags[0].name} {o.tags[0].confidence:.0%}")
                        for o in objects
                    ]
                    st.image(
                        draw_boxes(image_bytes, boxes),
                        alt="Image with detected objects outlined",
                    )
                else:
                    st.caption("No objects found.")
            elif name == "People":
                if people:
                    boxes = [(p.bounding_box, f"{p.confidence:.0%}") for p in people]
                    st.image(
                        draw_boxes(image_bytes, boxes),
                        caption=f"{len(people)} people (confidence ≥ 50%)",
                        alt="Image with detected people outlined",
                    )
                else:
                    st.caption("No people found.")
            elif lines:
                st.code("\n".join(lines), language=None)
            else:
                st.caption("No text found.")


client = get_client()

st.title(":material/image_search: Image analysis")
st.caption("Extract captions, tags, objects, people and text from any image with Azure AI Vision.")

if client is None:
    st.error(
        "Set `VISION_KEY` and `VISION_ENDPOINT` in a local `.env` file, "
        "or as App Settings on Azure.",
        title="Missing configuration",
    )
    st.stop()

image_bytes: bytes | None = None

with st.sidebar:
    st.header(":material/tune: Settings")
    source = st.segmented_control(
        "Image source", ["Web URL", "Upload"], default="Web URL", key="source"
    )
    if source == "Upload":
        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp", "gif", "webp"]
        )
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
    elif source == "Web URL":
        image_url = st.text_input("Direct image URL", value=DEFAULT_URL).strip()
        if image_url:
            try:
                image_bytes = download_image(image_url)
            except Exception as exc:
                st.error(f"Could not load the image: {exc}", title="Invalid image URL")

    selected = st.pills(
        "Visual features",
        list(FEATURES),
        selection_mode="multi",
        default=["Caption", "Tags"],
        key="features",
    )
    analyse = st.button(
        "Analyse image",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=image_bytes is None or not selected,
    )

image_id = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None

if analyse:
    try:
        with st.spinner("Analysing..."):
            st.session_state.analysis = {
                "image_id": image_id,
                "result": client.analyze(
                    image_data=image_bytes,
                    visual_features=[FEATURES[name][0] for name in selected],
                ),
            }
    except Exception as exc:
        st.session_state.pop("analysis", None)
        st.error(str(exc), title="Analysis failed")

image_col, results_col = st.columns(2, gap="large")

with image_col.container(border=True):
    st.markdown("#### :material/image: Image")
    if image_bytes is None:
        st.info("Pick an image in the sidebar: paste a URL or upload a file.")
    else:
        st.image(image_bytes, alt="Image to analyse", width="stretch")

with results_col.container(border=True):
    st.markdown("#### :material/query_stats: Results")
    analysis = st.session_state.get("analysis")
    if analysis and analysis["image_id"] == image_id:
        show_results(analysis["result"], image_bytes)
    else:
        st.info("Choose the features you want, then click **Analyse image**.")
