import asyncio
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaPlayer
import json
import os

import cv2
from av import VideoFrame
from aiortc import VideoStreamTrack

ROOT = os.path.dirname(__file__)

pc_set = set()


class MacWebcamTrack(VideoStreamTrack):
    def __init__(self, device_index=0, width=640, height=480):
        super().__init__()
        self.cap = cv2.VideoCapture(device_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[
        RTCIceServer(urls="stun:stun.l.google.com:19302") # Optional: Use a STUN server for connectivity
    ]))
    pc_set.add(pc)

    # --- Video Track ---
    # Use a webcam (adjust device name if necessary, default 0 works for most)
    # The aiortc docs provide more advanced ways to handle media using GStreamer if needed

    pc.addTrack(MacWebcamTrack(device_index=0))
    

    # --- Data Channel ---
    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            print(f"Message from data channel: {message}")
            if isinstance(message, str) and message == "ping":
                channel.send("pong")
    
    # Event handler for connection state changes
    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed" or pc.connectionState == "closed":
            await pc.close()
            pc_set.discard(pc)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def on_shutdown(app):
    # Close all peer connections on shutdown
    for pc in list(pc_set):
        await pc.close()
    pc_set.clear()

if __name__ == "__main__":
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    web.run_app(app, port=8080)
