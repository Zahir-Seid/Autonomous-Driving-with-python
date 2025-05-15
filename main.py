import socketio
import numpy as np
import asyncio
from fastapi import FastAPI
from keras.models import load_model # type: ignore
from keras.losses import MeanSquaredError # type: ignore
from PIL import Image
import base64
from io import BytesIO
from Augmentation_Techniques import ImageAugmentor

# Initialize the augmentation class
augmentor = ImageAugmentor()

# ASGI-compatible SocketIO server
sio = socketio.AsyncServer(async_mode='asgi')
app = FastAPI()
socket_app = socketio.ASGIApp(sio, other_asgi_app=app)

# Load model once during startup for efficiency
model = load_model('model/model.h5', custom_objects={'mse': MeanSquaredError()})
speed_limit = 10

# Async event handler for client connection
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    await send_control(0, 0)

# Async event handler for telemetry data
@sio.event
async def telemetry(sid, data):
    speed = float(data['speed'])
    image_data = data['image']

    # Decode image and preprocess asynchronously
    image = await asyncio.get_event_loop().run_in_executor(None, decode_and_process_image, image_data)
    steering_angle = await asyncio.to_thread(predict_steering_angle, image)
    throttle = 1.0 - speed / speed_limit

    print(f"Steering: {steering_angle:.4f}, Throttle: {throttle:.4f}, Speed: {speed:.2f}")
    await send_control(steering_angle, throttle)

# Decode base64 image and preprocess using augmentor class
def decode_and_process_image(image_data):
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = np.asarray(image)
    return augmentor.preprocess(image)

# Predict steering angle using the preloaded model
def predict_steering_angle(image):
    image = np.array([image])
    return float(model.predict(image, verbose=0))

# Send control to the client
async def send_control(steering_angle, throttle):
    await sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Entry point
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=4567, workers=4, log_level="info")

# uvicorn main:socket_app --host 0.0.0.0 --port 4567 --ws websockets

