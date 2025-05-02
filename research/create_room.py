import time
from livekit import api
import livekit.api.access_token as access_token
import livekit.api.ingress as ingress
import livekit.api.models as models

# Configuration
LIVEKIT_HOST = "https://zeeshan-60mort1m.livekit.cloud"
LIVEKIT_API_KEY = "VDZVIa4nqfx6q04ErUGZ0gIa2MBekvvAU6wgqXZ4Tb1"
LIVEKIT_API_SECRET = "APIiywuu7CBoUzF"
ROOM_NAME = "my-python-room"
PARTICIPANT_IDENTITY = "python-user"
INGRESS_NAME = "my-python-ingress"

# Initialize LiveKit clients
room_service_client = api.LiveKitAPI(api_key=LIVEKIT_API_KEY,api_secret=LIVEKIT_API_SECRET,url=LIVEKIT_HOST)
ingress_client = room_service_client.ingress


def create_room_if_not_exists(room_name):
    """
    Creates a room if it does not already exist.

    Args:
        room_name (str): The name of the room to create.
    """
    try:
        room = room_service_client.get_room(room_name)
        print(f"Room '{room_name}' already exists.")
        return room
    except Exception:  #  Catch the exception that get_room raises when the room does not exist
        try:
            room = room_service_client.create_room(name=room_name)
            print(f"Room '{room_name}' created.")
            return room
        except Exception as e:
            print(f"Error creating room: {e}")
            return None



def create_access_token(room_name, identity):
    """
    Generates an Access Token for a participant.

    Args:
        room_name (str): The name of the room.
        identity (str): The identity of the participant.

    Returns:
        str: The generated JWT token, or None on error.
    """
    try:
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, identity=identity)
        token.add_grant(VideoGrant(room_name=room_name, room_join=True, ingress_publish=True))  # Added ingress_publish
        jwt = token.to_jwt()
        print(f"Generated JWT for identity '{identity}': {jwt}")
        return jwt
    except Exception as e:
        print(f"Error generating JWT: {e}")
        return None



def create_ingress(room_name, identity):
    """
    Creates an ingress for streaming into the specified room.

    Args:
        room_name (str): The name of the room.
        identity (str): Participant identity for the ingress
    Returns:
        dict: Information about the created ingress, or None on error.
    """
    try:
        ingress = ingress_client.create(
            name=INGRESS_NAME,  # Add a name for the ingress
            room_name=room_name,
            participant_identity=identity, # Set Participant Identity
            participant_name="Ingress Stream",
            input_type=IngressInput.STREAM,  # Use STREAM for RTMP
            audio=IngressAudioOptions(bitrate=128000),
            video=IngressVideoOptions(
                bitrate=3000000,
                layers=[
                    {"quality": "high", "width": 1280, "height": 720, "bitrate": 2000000},
                    {"quality": "medium", "width": 640, "height": 360, "bitrate": 1000000},
                    {"quality": "low", "width": 320, "height": 180, "bitrate": 500000},
                ],
            ),
        )
        print(f"Ingress created: {ingress}")
        return {
            "ingress_id": ingress.ingress_id,
            "stream_key": ingress.stream_key,
            "url": ingress.url,
            "status": ingress.status,
        }
    except Exception as e:
        print(f"Error creating ingress: {e}")
        return None



def get_ingress_info(ingress_id):
    """
    Retrieves the ingress info.

    Args:
      ingress_id: id of the ingress

    Returns:
      Ingress info
    """
    try:
        ingress = ingress_client.get(ingress_id)
        return {
            "ingress_id": ingress.ingress_id,
            "stream_key": ingress.stream_key,
            "url": ingress.url,
            "status": ingress.status
        }
    except Exception as e:
        print(f"Error getting ingress info: {e}")
        return None

def delete_ingress(ingress_id):
    """
    Deletes the ingress.

    Args:
      ingress_id: id of the ingress to delete
    """
    try:
        ingress_client.delete(ingress_id)
        print(f"Ingress {ingress_id} deleted")
    except Exception as e:
        print(f"Error deleting ingress: {e}")



if __name__ == "__main__":
    # 1. Create a room (or get it if it exists)
    room = create_room_if_not_exists(ROOM_NAME)
    if room is None:
        exit(1)  # Exit if room creation failed.

    # 2. Generate a JWT for a participant
    token = create_access_token(ROOM_NAME, PARTICIPANT_IDENTITY)
    if token is None:
        exit(1)  # Exit if token generation failed.

    # 3. Create an ingress for streaming
    ingress_info = create_ingress(ROOM_NAME, PARTICIPANT_IDENTITY)
    if ingress_info is None:
        exit(1)  # Exit if ingress creation failed.

    # Print the important information
    print("---------------------------------------------------")
    print(f"Room Name: {ROOM_NAME}")
    print(f"Participant Identity: {PARTICIPANT_IDENTITY}")
    print(f"JWT Token: {token}")
    print("---------------------------------------------------")
    print("Ingress Information:")
    print(f"Ingress ID: {ingress_info['ingress_id']}")
    print(f"Stream Key: {ingress_info['stream_key']}")
    print(f"RTMP URL: {ingress_info['url']}")
    print(f"Ingress Status: {ingress_info['status']}")
    print("---------------------------------------------------")
    print("OBS Studio Setup Instructions:")
    print("1. Open OBS Studio.")
    print("2. Go to Settings > Stream.")
    print("3. Select \"Custom...\" as the Service.")
    print(f"4. Enter the following:")
    print(f"   - URL: {ingress_info['url']}")
    print(f"   - Stream Key: {ingress_info['stream_key']}")
    print("5. Click OK.")
    print("6. Start streaming from OBS Studio.")
    print("---------------------------------------------------")

    # Example of getting ingress info
    ingress_id = ingress_info['ingress_id']
    if ingress_id:
        ingress_details = get_ingress_info(ingress_id)
        print(f"Retrieved Ingress Details: {ingress_details}")

    # Keep the script running to allow the stream to start.  In a real application,
    # you might have other logic here, like a web server that provides this
    # information to a client.  For this example, we'll just sleep.
    input("Press Enter to delete the ingress and exit...")

    # Clean up the ingress when done.
    if ingress_info and ingress_info['ingress_id']:
        delete_ingress(ingress_info['ingress_id'])
    print("Exiting...")






import livekit.api.access_token as access_token
import livekit.api.ingress as ingress
import livekit.api.models as models
from livekit import api

# Server details
LIVEKIT_HOST = "YOUR_LIVEKIT_HOST"
LIVEKIT_API_KEY = "YOUR_LIVEKIT_API_KEY"
LIVEKIT_API_SECRET = "YOUR_LIVEKIT_API_SECRET"

# Create an API client
lk_api = api.LiveKitAPI(LIVEKIT_HOST, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
ingress_service = lk_api.ingress

# Configure Ingress settings
room_name = "my-room"
ingress_input = ingress.IngressInput.RTMP_INPUT
audio_encoding = ingress.IngressAudioEncoding.OPUS_ENCODING
video_encoding = ingress.IngressVideoEncoding.H264_ENCODING
preset = ingress.IngressPreset.STANDARD_PRESET

# Create Ingress request
create_ingress_request = ingress.CreateIngressRequest(
    room_name=room_name,
    participant_identity="ingress-participant",
    participant_name="Ingress Participant",
    input_type=ingress_input,
    audio=ingress.IngressAudioOptions(encoding=audio_encoding),
    video=ingress.IngressVideoOptions(encoding=video_encoding, preset=preset)
)

# Create the Ingress
try:
    ingress_response = await ingress_service.create_ingress(create_ingress_request)
    print(f"Ingress created successfully. Ingress ID: {ingress_response.ingress_id}")
    print(f"Streaming URL: {ingress_response.url}")
    print(f"Stream Key: {ingress_response.stream_key}")

except Exception as e:
     print(f"Error creating ingress: {e}")