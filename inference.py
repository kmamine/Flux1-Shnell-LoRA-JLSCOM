from diffusers import AutoPipelineForText2Image
import torch

# prompt template used to totrain the model,  [JLSCOM] is the trigger word that is used to create the specific style
prompt  = """[JLSCOM] Garment Type: Slim-Fit Jeans Fit and 
Style: Slim-fit, designed to hug the legs closely without being overly tight, offering a contemporary, streamlined appearance. 
Color and Pattern: Soft pastel green in a solid shade, adding a subtle pop of color to outfits while maintaining a minimalist, modern look. 
Fabric/Material: Crafted from a stretch cotton blend, providing comfort, flexibility, and durability. Details: Traditional five-pocket design with two front pockets, two back pockets, and a small coin pocket, all seamlessly integrated for functionality and style. Display Style: Displayed in a flat lay to highlight the overall structure and color. 
Background and Lighting: Set against a light gray background with soft, even lighting to bring out the pastel hue of the jeans without overshadowing it. Shape: Fitted shape with a tapered leg, maintaining a sleek and tailored silhouette from hip to ankle. 
Closures: Secured with a standard button and zipper fly in matching tones for a seamless look. Branding: Minimal branding with a discreet internal label; no external logos, maintaining a clean, understated aesthetic. 
Cuffs and Hems: Clean, stitched hems at the ankle, allowing the jeans to be worn full-length or slightly rolled for a casual look. 
Fit: Slim yet comfortable, allowing ease of movement while staying fitted through the legs. 
Length: Full length, designed to sit right at the ankle, suitable for pairing with both casual and semi-formal footwear. Occasion: Versatile enough for both casual daily wear and smart-casual occasions, adding a fresh twist to any wardrobe. 
Style Influence: Inspired by modern minimalist fashion, with a focus on clean lines and a refined color palette. Seasonal Suitability: Ideal for spring and summer wear due to the light color and breathable fabric. Texture: Smooth, soft finish with a hint of stretch, ensuring comfort during prolonged wear. 
Weight: Medium weight, suitable for warm weather without feeling too thin. Finish: Matte finish, enhancing the soft, pastel tone for a polished, sophisticated look. Aesthetic Style: Casual chic, blending comfort with a contemporary style that is effortlessly versatile. Target Audience: Suitable for individuals seeking stylish yet comfortable jeans with a unique color that is easy to style. 
Ease of Care: Machine washable, with colorfastness to retain the pastel shade after multiple washes."""

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-schnell', 
                                                     torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('Amine-CV/JLSCOM_garment_LoRA_flux_schnell_v1', 
                           weight_name='JLSCOM_garment_LoRA_flux_schnell.safetensors')

image = pipeline(prompt).images[0]
image.save("my_image.png")
