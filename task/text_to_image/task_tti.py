import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        for attachment in attachments:
            if attachment.type and attachment.type == 'image/png':
                image_data = await bucket_client.get_file(attachment.url)
                filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(filename, 'wb') as f:
                    f.write(image_data)
                print(f"Image saved: {filename}")


def start() -> None:
    # 1. Create DialModelClient instance
    dalle_client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name='dall-e-3',
        api_key=API_KEY
    )

    # 2. Set user input
    user_input = 'Sunny day on Bali'
    
    # 3. Generate image
    print(f"Generating image for: {user_input}...")
    ai_message = dalle_client.get_completion(
        messages=[Message(role=Role.USER, content=user_input)]
    )

    # 4. Check for attachments and save
    if (custom_content := ai_message.custom_content) and (attachments := custom_content.attachments):
        asyncio.run(_save_images(attachments))

    # 5. Try to configure the picture for output via `custom_fields`
    print(f"Generating HD vivid image for: {user_input}...")
    ai_message_custom = dalle_client.get_completion(
        messages=[Message(role=Role.USER, content=user_input)],
        custom_fields={"size": Size.square, "style": Style.vivid, "quality": Quality.hd}
    )
    if (custom_content := ai_message_custom.custom_content) and (attachments := custom_content.attachments):
        asyncio.run(_save_images(attachments))

    # 6. Test it with the 'imagegeneration@005' (Google image generation model)
    print(f"Generating image with Google model (imagegeneration@005) for: {user_input}...")
    google_client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name='imagegeneration@005',
        api_key=API_KEY
    )
    ai_message_google = google_client.get_completion(
        messages=[Message(role=Role.USER, content=user_input)]
    )
    if (custom_content := ai_message_google.custom_content) and (attachments := custom_content.attachments):
        asyncio.run(_save_images(attachments))


start()
