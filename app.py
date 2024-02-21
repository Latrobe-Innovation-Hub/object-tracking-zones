"""
Edited
Author: Andrew J McDonald
Date: 17.07.2023
"""

import streamlit as st
from obj_det_and_trk import *
#from obj_det_and_trk_zones_streamlit import *
import tempfile

import streamlit as st
#import redis
import json

import torch
import threading




rtsp_feeds = {
    "CollaborationHub201C": "rtsp://192.168.3.75:9000/live",
    "Kitchen": "rtsp://192.168.3.77:9000/live",
    "OpenDeskSpace203": "rtsp://192.168.3.87:9000/live",
    "MV12-EntryDoor": "rtsp://192.168.3.85:9000/live",
    "PitchSpace213": "rtsp://192.168.3.69:9000/live",
    "CoWorkLounge201B": "rtsp://192.168.3.78:9000/live",
}

stored_videos = {
    "video-1": "videos/crowd-1.mp4",
    "video-2": "videos/crowd-2.mp4",
    "video-3": "videos/test.mp4",
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
                Occupancy Utilisation Demo</h1>
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

# Initialize or update the tracking state
if 'tracking_started' not in st.session_state:
    st.session_state.tracking_started = False
    
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

    inference_msg = st.empty()
    st.sidebar.title("Configuration")
    
    input_source = st.sidebar.radio("Source", ('WebCam', 'Video', 'RTSP',))
    
    if input_source == "RTSP":
        feed_selection = st.sidebar.selectbox("Select RTSP Feed", list(rtsp_feeds.keys()))
        source = rtsp_feeds[feed_selection]
    elif input_source == "Video":
        video_selection = st.sidebar.selectbox("Select Video", list(stored_videos.keys()))
        source = stored_videos[video_selection]
    elif input_source == "WebCam":
        webcam_src = {
            'webcam-0': 0,
            'webcam-1': 1,
            'webcam-2': 2,
            'webcam-3': 3,
            'webcam-4': 4,
        }
        webcam_choice = st.sidebar.selectbox("Select Webcam Source", list(webcam_src.keys()))
        source = str(webcam_src[webcam_choice])
        
    device_choice = st.sidebar.selectbox("Set Inference Device", ('CPU', 'GPU')) 
    device = None
    if device_choice == 'GPU':
        device = '0'
    else:
        device = 'cpu'
    
    model_choice = st.sidebar.selectbox("Choose Model Size", list(weights_dict.keys()))
    weights = weights_dict[model_choice]
    
    conf_thres = st.sidebar.text_input("Set Detection Confidence Level", "0.5")
    
    blacked_choice = st.sidebar.radio("Show Blacked Out Frame?", ('Yes', 'No'))

    save_output_video = 'No'                                     

    if save_output_video == 'Yes':
        nosave = False
        display_labels=False
   
    else:
        nosave = True
        display_labels = True 
        
    # Button to toggle tracking on and off
    if st.sidebar.button("Start/Stop Tracking"):
        st.session_state.tracking_started = not st.session_state.tracking_started

    # Now check if tracking should be started or stopped based on the session_state
    if st.session_state.tracking_started:
        column1, column2 = st.columns(2)
        stframe = column1.empty()
        stframe2 = column2.empty()
        
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
            
        if blacked_choice == 'Yes':
            blacked=True
        else:
            blacked=None
            
        detect(weights=weights, 
               source=source,
               stframe=stframe,
               stframe2=stframe2,
               kpi5_text=kpi5_text,
               kpi6_text=kpi6_text,
               kpi7_text=kpi7_text,
               conf_thres=float(conf_thres),
               device=device,
               classes=0, nosave=nosave, 
               display_labels=display_labels,
               blacked=blacked)

    else:
        # Code to clear the output or handle the UI when tracking is stopped
        # This could be clearing the placeholders or showing a message
        inference_msg.info("Tracking is not active.")
         
    torch.cuda.empty_cache()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
