import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
import tempfile

def main():
    st.title('Smart Traffic Solutions')
    html_temp="""
    <div style="background-color:green;padding:10px">
    <h2 style="color:white;text-align:center;">Smart Traffic Solution</h2>
    </div>
    """
    #img=Image.open("download.jpg")
    img=cv2.imread("download.jpg")
    #st.image(
    #    img,
    #    caption="Image of a lambo",
    ##    width=200,
     #   channels="BGR"
    #)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Use markdown with CSS to center the image
        st.markdown(
            """
            <style>
            .center {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Embed the image in a centered div
        st.markdown(
            f"""
            <div class="center">
                <img src="data:image/jpeg;base64,{st.image(img_rgb, use_container_width=True)}" alt="Image of a lambo" style="width:50px;">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("Failed to load the image.")

    st.markdown(html_temp,unsafe_allow_html=True)
    video_file = st.file_uploader("Upload the clip", type=["mp4","avi","mov"])
    if video_file is not None:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_file.read())
            video_file_path=f.name

        st.video(video_file_path)

    #predict(video_file,variance,skewness,curotosis,entropy)
        cap=cv2.VideoCapture(video_file_path)
        if cap.isOpened()==False:
            st.write("Error opening video stream or file")
            return

        st.write("Processing frames...")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        st.write(f"Video Resolution: {frame_width}x{frame_height}") 

        frame_count=0

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                st.image(frame, caption=f"Frame {frame_count}", use_container_width=True)
                frame_count +=1
                if frame_count > 10:
                    
                    break


                #if cv2.waitKey(25) & 0xFF == ord('q'):
                #    break
            else:
                break
    
    
        cap.release()
    #cv2.destroyAllWindows()


    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curotosis,entropy) 
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

    st.header("Visualization Dashboard")
    st.markdown(
        '<iframe src="http://127.0.0.1:8050" width="100%" height="600"></iframe>',
        unsafe_allow_html=True
    )
    

if __name__=='__main__':
    main()
    