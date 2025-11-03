import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
import skimage.draw as ski
from io import BytesIO

st.set_page_config(page_title="Face Morphing App", layout="wide")


for key, default in {
    "points1": [],
    "points2": [],
    "last_img1_click": None,
    "last_img2_click": None,
    "force_redraw_left": 0,
    "force_redraw_right": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

col_upload1, col_upload2 = st.columns(2)
with col_upload1:
    uploaded1 = st.file_uploader("Upload Left Image", type=["jpg", "jpeg", "png"], key="left_upload")
with col_upload2:
    uploaded2 = st.file_uploader("Upload Right Image", type=["jpg", "jpeg", "png"], key="right_upload")

if not (uploaded1 and uploaded2):
    st.warning("Please upload both images to start.")
    st.stop()


img1 = Image.open(uploaded1).convert("RGB")
img2 = Image.open(uploaded2).convert("RGB")

target_width = 600
w1, h1 = img1.size
h_new = int((h1 / w1) * target_width)
img1 = img1.resize((target_width, h_new))
img2 = img2.resize((target_width, h_new))


def draw_points(image, points):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    font = ImageFont.load_default()
    for i, pt in enumerate(points):
        x, y = pt["x"], pt["y"]
        draw.text((x, y), str(i + 1), fill="red", font=font)
    return img_copy

col1, col2 = st.columns(2)
with col1:
    clicked1 = streamlit_image_coordinates(
        draw_points(img1, st.session_state.points1),
        key=f"img1_click_{st.session_state['force_redraw_left']}"
    )
with col2:
    clicked2 = streamlit_image_coordinates(
        draw_points(img2, st.session_state.points2),
        key=f"img2_click_{st.session_state['force_redraw_right']}"
    )

def handle_click(image_key, clicked, last_click_key, redraw_key):
    if clicked and clicked != st.session_state[last_click_key]:
        coords = (clicked["x"], clicked["y"])
        existing = [(p["x"], p["y"]) for p in st.session_state[image_key]]
        if coords not in existing:
            st.session_state[image_key].append(clicked)
            st.session_state[last_click_key] = clicked
            st.session_state[redraw_key] += 1
            st.rerun()

handle_click("points1", clicked1, "last_img1_click", "force_redraw_left")
handle_click("points2", clicked2, "last_img2_click", "force_redraw_right")


with st.expander("Show All Points"):
    st.write("**Left Image Points:**", st.session_state.points1)
    st.write("**Right Image Points:**", st.session_state.points2)

# ---- Morphing helper functions ----
def computeAffine(tri1_pts, tri2_pts):
    ones = [1 for _ in range(3)]
    firstTriResults = np.array([tri1_pts[:,0], tri1_pts[:,1], ones])
    secTriResults = np.array([tri2_pts[:,0], tri2_pts[:,1], ones])
    return np.matmul(secTriResults, np.linalg.pinv(firstTriResults))

def changeShape(imm1, dul, curCoord, newCoord):
    output = np.zeros_like(imm1)
    for i in range(len(dul.simplices)):
        tri_indices = dul.simplices[i]
        trans = computeAffine(curCoord[tri_indices], newCoord[tri_indices])
        invTrans = np.linalg.pinv(trans)

        rr, cc = ski.polygon(newCoord[tri_indices][:,1], newCoord[tri_indices][:,0])
        valid_mask = (
            (rr >= 0) & (rr < imm1.shape[0]) &
            (cc >= 0) & (cc < imm1.shape[1])
        )
        rr, cc = rr[valid_mask], cc[valid_mask]

        outputPoints = np.stack([cc, rr, np.ones(len(rr))])
        interpolatedCoords = invTrans @ outputPoints
        xi = np.column_stack((interpolatedCoords[0], interpolatedCoords[1]))

        pr, pc = ski.polygon(curCoord[tri_indices][:,1], curCoord[tri_indices][:,0])
        valid_mask2 = (
            (pr >= 0) & (pr < imm1.shape[0]) &
            (pc >= 0) & (pc < imm1.shape[1])
        )
        pr, pc = pr[valid_mask2], pc[valid_mask2]
        points = np.column_stack((pc, pr))
        values = imm1[pr, pc]

        interpolatedVals = griddata(points, values, xi, method='nearest', fill_value=0)
        output[rr, cc] = interpolatedVals
    return output

def morph(im1, im2, im1_pts, im2_pts, tri, shape_frac, color_frac):
    midPoints = shape_frac * im1_pts + (1 - shape_frac) * im2_pts
    im1_morphed = changeShape(im1, tri, im1_pts, midPoints)
    im2_morphed = changeShape(im2, tri, im2_pts, midPoints)
    combined = color_frac * im1_morphed + (1 - color_frac) * im2_morphed
    return combined

st.markdown("---")
st.subheader("Morphing Controls")

col_controls = st.columns(2)
with col_controls[0]:
    shape_frac_str = st.text_input("Shape percent of image 1 (between 0 and 1)", value="0.5")
with col_controls[1]:
    color_frac_str = st.text_input("Color percent of image 1(between 0 and 1)", value="0.5")

def read_decimal(s):
    try:
        val = float(s)
        if 0.0 <= val <= 1.0:
            return val
        else:
            st.error("Value must be between 0 and 1.")
            return None
    except ValueError:
        st.error("Please enter a valid number between 0 and 1.")
        return None

shape_frac = read_decimal(shape_frac_str)
color_frac = read_decimal(color_frac_str)

# ---- Morph button ----
if st.button(" Morph Faces"):
    if shape_frac is not None and color_frac is not None:
        pts1 = np.array([[p["x"], p["y"]] for p in st.session_state.points1], dtype=np.float32)
        pts2 = np.array([[p["x"], p["y"]] for p in st.session_state.points2], dtype=np.float32)

        w, h = img1.size
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        pts1 = np.vstack([pts1, corners])
        pts2 = np.vstack([pts2, corners])

        if len(pts1) == len(pts2) and len(pts1) >= 3:
            tri = Delaunay(0.5 * (pts1 + pts2))
            im1_np = np.array(img1).astype(np.float32)
            im2_np = np.array(img2).astype(np.float32)

            morphed = morph(im1_np, im2_np, pts1, pts2, tri, shape_frac, color_frac)
            morphed = np.clip(morphed, 0, 255).astype(np.uint8)

           
            morphed_img = Image.fromarray(morphed.astype(np.uint8))
            morphed_img = morphed_img.resize(img1.size)

            st.image(morphed_img, caption=f"Morphed Result (shape={shape_frac}, color={color_frac})", use_container_width=True)

            buf = BytesIO()
            morphed_img.save(buf, format="PNG")
            st.download_button(
                label="Download Morphed Image",
                data=buf.getvalue(),
                file_name="morphed_result.png",
                mime="image/png"
            )
        else:
            st.error("Both images must have the same number of corresponding points (>=3).")
