"""
Edited
Author: Andrew J McDonald
Date: 17.07.2023
"""

import streamlit as st
#from obj_det_and_trk_streamlit import *
from obj_det_and_trk_zones_streamlit import *
import tempfile

import streamlit as st
#import redis
import json

import torch

# Connect to Redis
#r = redis.Redis(host='localhost', port=6379, db=0)

# Read the data
#zone_counts = json.loads(r.get('zone_counts'))



# Example RTSP feed URLs
rtsp_feeds = {
    "CollaborationHub201C": "rtsp://192.168.3.75:9000/live",
    "Kitchen": "rtsp://192.168.3.77:9000/live",
    "OpenDeskSpace203": "rtsp://192.168.3.87:9000/live",
    "MV12-EntryDoor": "rtsp://192.168.3.85:9000/live",
    "PitchSpace213": "rtsp://192.168.3.69:9000/live",
    "CoWorkLounge201B": "rtsp://192.168.3.78:9000/live",
}

# Define YOLO model weights
weights_dict = {
    'nano': 'yolov5n.pt',
    'small': 'yolov5s.pt',
    'medium': 'yolov5m.pt',
    'large': 'yolov5l.pt',
    'extra_large': 'yolov5x.pt',
}

#--------------------------------Web Page Designing------------------------------
hide_menu_style = """
    <style>
        MainMenu {visibility: hidden;}
        
        
         div[data-testid="stHorizontalBlock"]> div:nth-child(1)
        {  
            border : 2px solid #doe0db;
            border-radius:5px;
            text-align:center;
            color:black;
            background:dodgerblue;
            font-weight:bold;
            padding: 25px;
            
        }
        
        div[data-testid="stHorizontalBlock"]> div:nth-child(2)
        {   
            border : 2px solid #doe0db;
            background:dodgerblue;
            border-radius:5px;
            text-align:center;
            font-weight:bold;
            color:black;
            padding: 25px;
            
        }
        
        div[data-testid="stHorizontalBlock"]> div:nth-child(3)
        {   
            border : 2px solid #doe0db;
            background:dodgerblue;
            border-radius:5px;
            text-align:center;
            font-weight:bold;
            color:black;
            padding: 25px;
            
        }
    </style>
    """

main_title = """
            <div>
                <h1 style="color:black;
                text-align:center; font-size:35px;
                margin-top:-95px;">
                YOLOv5 People Detection and Tracking</h1>
            </div>
            """
    
#sub_title = """
#            <div>
#                <h6 style="color:dodgerblue;
#                text-align:center;
#                margin-top:-40px;">
#                Streamlit Basic Dasboard </h6>
#            </div>
#            """
#--------------------------------------------------------------------------------


#---------------------------Main Function for Execution--------------------------
def main():
    st.set_page_config(page_title='Dashboard', 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')
    
    st.markdown(hide_menu_style, 
                unsafe_allow_html=True)

    st.markdown(main_title,
                unsafe_allow_html=True)
                
    kpi5, kpi6, kpi7 = st.columns(3)

    #st.markdown(sub_title,unsafe_allow_html=True)

    inference_msg = st.empty()
    st.sidebar.title("Configuration")
    
    input_source = st.sidebar.radio("Source", ('RTSP Feed', 'Video', 'WebCam'))
    #input_source = 'RTSP Feed'
    
    model_choice = st.sidebar.selectbox("Choose YOLO Model Size", list(weights_dict.keys()))

    # Use the selection to get the corresponding weights file
    weights = weights_dict[model_choice]
    
    conf_thres = st.sidebar.text_input("Conf", "0.25")
    #conf_thres = "0.25"

    #save_output_video = st.sidebar.radio("Save output video?",
    #                                     ('Yes', 'No'))
    save_output_video = 'No'                                     

    if save_output_video == 'Yes':
        nosave = False
        display_labels=False
   
    else:
        nosave = True
        display_labels = True 
           
    #weights = "yolov5n.pt"
    device="0"
    
    # ------------------------- RTSP VIDEO ------------------------
    if input_source == "RTSP Feed":
        # Let the user select an RTSP feed
        feed_selection = st.sidebar.selectbox("Select RTSP Feed", list(rtsp_feeds.keys()))

        # Get the selected RTSP URL
        rtsp_url = rtsp_feeds[feed_selection]

        if st.sidebar.button("Start Tracking"):
            stframe = st.empty()
            
            #st.markdown("""<h4 style="color:black;">
            #                Memory Overall Statistics</h4>""", 
            #                unsafe_allow_html=True)
            #kpi5, kpi6, kpi7 = st.columns(3)
            
            with kpi5:
                st.markdown("""<h5 style="color:black;">
                            CPU Utilization</h5>""", 
                            unsafe_allow_html=True)
                kpi5_text = st.markdown("--")
            
            with kpi6:
                st.markdown("""<h5 style="color:black;">
                            Memory Usage</h5>""", 
                            unsafe_allow_html=True)
                kpi6_text = st.markdown("--")
                
            with kpi7:
                st.markdown("""<h5 style="color:black;">
                            Detections Observed</h5>""", 
                            unsafe_allow_html=True)
                kpi7_text = st.markdown("--")

            # Call the detect function with the RTSP URL
            detect(weights=weights, 
                   source=rtsp_url,  # Use the selected RTSP URL
                   stframe=stframe, 
                   kpi5_text=kpi5_text,
                   kpi6_text=kpi6_text,
                   kpi7_text=kpi7_text,
                   conf_thres=float(conf_thres),
                   device="0",
                   classes=0, nosave=nosave, 
                   display_labels=display_labels)
                   
            # Attempt to read the data
            try:
                with open('zone_counts.json', 'r') as f:
                    zone_counts = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                zone_counts = {}  # Use an empty dictionary or some default value if the file is missing or empty

            # Display in Streamlit
            for zone, details in zone_counts.items():
                st.header(f"{zone}: {details['object_count']} objects")

            inference_msg.success("Inference Complete!")

    # ------------------------- LOCAL VIDEO ------------------------
    if input_source == "Video":
        
        #video = st.sidebar.file_uploader("Select input video", 
        #                                type=["mp4", "avi"], 
        #                                accept_multiple_files=False)
                                        
        video = st.sidebar.selectbox("Select Video", ("videos/crowd-1.mp4","videos/crowd-2.mp4","videos/test.mp4"))
                                        
        #if video is not None and st.sidebar.button("Start Tracking"):
            # Save the uploaded video to a temporary file
        #    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        #        tmpfile.write(video.read())
        #        video_path = tmpfile.name
        
        if st.sidebar.button("Start Tracking"):
            
            stframe = st.empty()
            
            st.markdown("""<h4 style="color:black;">
                            Memory Overall Statistics</h4>""", 
                            unsafe_allow_html=True)
            kpi5, kpi6 = st.columns(2)

            with kpi5:
                st.markdown("""<h5 style="color:black;">
                            CPU Utilization</h5>""", 
                            unsafe_allow_html=True)
                kpi5_text = st.markdown("0")
            
            with kpi6:
                st.markdown("""<h5 style="color:black;">
                            Memory Usage</h5>""", 
                            unsafe_allow_html=True)
                kpi6_text = st.markdown("0")
            
            detect(weights=weights, 
                   source=video,  
                   stframe=stframe, 
                   kpi5_text=kpi5_text,
                   kpi6_text = kpi6_text,
                   conf_thres=float(conf_thres),
                   device="cpu",
                   classes=0,nosave=nosave, 
                   display_labels=display_labels)

            inference_msg.success("Inference Complete!")



    # ------------------------- LOCAL VIDEO ------------------------
    if input_source == "WebCam":
        webcam_src = {
            'webcam-0': 0,
            'webcam-1': 1,
            'webcam-2': 2,
            'webcam-3': 3,
            'webcam-4': 4,
        }
        
        webcam_choice = st.sidebar.selectbox("webcam", list(webcam_src.keys()))

        # Use the selection to get the corresponding weights file
        webcam = webcam_src[webcam_choice]
        
        if st.sidebar.button("Start Tracking"):
            
            stframe = st.empty()
            
            st.markdown("""<h4 style="color:black;">
                            Memory Overall Statistics</h4>""", 
                            unsafe_allow_html=True)
            kpi5, kpi6 = st.columns(2)

            with kpi5:
                st.markdown("""<h5 style="color:black;">
                            CPU Utilization</h5>""", 
                            unsafe_allow_html=True)
                kpi5_text = st.markdown("0")
            
            with kpi6:
                st.markdown("""<h5 style="color:black;">
                            Memory Usage</h5>""", 
                            unsafe_allow_html=True)
                kpi6_text = st.markdown("0")
            
            detect(weights=weights, 
                   source=f"{webcam}",  
                   stframe=stframe, 
                   kpi5_text=kpi5_text,
                   kpi6_text = kpi6_text,
                   conf_thres=float(conf_thres),
                   device="cpu",
                   classes=0,nosave=nosave, 
                   display_labels=display_labels)

            inference_msg.success("Inference Complete!")
           
    # --------------------------------------------------------------       
    torch.cuda.empty_cache()
    # --------------------------------------------------------------



# --------------------MAIN FUNCTION CODE------------------------
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
# ------------------------------------------------------------------
