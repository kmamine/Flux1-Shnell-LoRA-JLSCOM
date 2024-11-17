# Flux1-Shnell-LoRA-JLSCOM


This repo is to train `Flux1.0-Schnell` using a sample dataset from the Jules.com graments 

This model is licencend under `AGPL-3.0 Licence` 

Any commercial use of the code, configurations, or the model adapters is prohibited without an explicite written authorization. 

The model weights are hosted on `hugginface-hub` @ `Amine-CV/JLSCOM_garment_LoRA_flux_schnell_v1` and are under the `AGPL-3.0 Licence`.

---

## Inference and image generation

To generate images using this model adapters use the following code and replace `[trigger]` with `JLSCOM`:   

```python 
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('black-forest-labs/FLUX.1-schnell', torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('Amine-CV/JLSCOM_garment_LoRA_flux_schnell_v1', weight_name='JLSCOM_garment_LoRA_flux_schnell.safetensors')
image = pipeline('[trigger] Garment Type: Slim-Fit Jeans Fit and Style: Slim-fit, designed to hug the legs closely without being overly tight, offering a contemporary, streamlined appearance. Color and Pattern: Soft pastel green in a solid shade, adding a subtle pop of color to outfits while maintaining a minimalist, modern look. Fabric/Material: Crafted from a stretch cotton blend, providing comfort, flexibility, and durability. Details: Traditional five-pocket design with two front pockets, two back pockets, and a small coin pocket, all seamlessly integrated for functionality and style. Display Style: Displayed in a flat lay to highlight the overall structure and color. Background and Lighting: Set against a light gray background with soft, even lighting to bring out the pastel hue of the jeans without overshadowing it. Shape: Fitted shape with a tapered leg, maintaining a sleek and tailored silhouette from hip to ankle. Closures: Secured with a standard button and zipper fly in matching tones for a seamless look. Branding: Minimal branding with a discreet internal label; no external logos, maintaining a clean, understated aesthetic. Cuffs and Hems: Clean, stitched hems at the ankle, allowing the jeans to be worn full-length or slightly rolled for a casual look. Fit: Slim yet comfortable, allowing ease of movement while staying fitted through the legs. Length: Full length, designed to sit right at the ankle, suitable for pairing with both casual and semi-formal footwear. Occasion: Versatile enough for both casual daily wear and smart-casual occasions, adding a fresh twist to any wardrobe. Style Influence: Inspired by modern minimalist fashion, with a focus on clean lines and a refined color palette. Seasonal Suitability: Ideal for spring and summer wear due to the light color and breathable fabric. Texture: Smooth, soft finish with a hint of stretch, ensuring comfort during prolonged wear. Weight: Medium weight, suitable for warm weather without feeling too thin. Finish: Matte finish, enhancing the soft, pastel tone for a polished, sophisticated look. Aesthetic Style: Casual chic, blending comfort with a contemporary style that is effortlessly versatile. Target Audience: Suitable for individuals seeking stylish yet comfortable jeans with a unique color that is easy to style. Ease of Care: Machine washable, with colorfastness to retain the pastel shade after multiple washes.').images[0]
image.save("my_image.png")

```

### Trigger word 
You should use `JLSCOM` to trigger the image generation.

---

## Training 

Model trained with the help of [AI Toolkit by Ostris](https://github.com/ostris/ai-toolkit). 

The whole code used for training is availbele in this repo.

### Cloning repo
To retrain a model on your custom dataset:
```shell 
git clone https://github.com/kmamine/Flux1-Shnell-LoRA-JLSCOM.git
```
### Installing enviroment on conda
Use the provided `.yml` file to install the   conda `env`: 

```Shell
cd ./Flux1-Shnell-LoRA-JLSCOM/
conda env create -f env.yml
```

## Dataset preparation 

The training used 100 unlabled garment images from [Jules.com](Jules.com). 

<table>
    <tr>
        <td> <img src = "assests/image_0.jpg"> </td>
        <td> <img src = "assests/image_1.jpg"> </td>
        <td> <img src = "assests/image_2.jpg"> </td>
        <td> <img src = "assests/image_3.jpg"> </td>
    </tr>
</table>

To create high quality captions we used the `OpenAI GPT4-o` model API. 

```python 
import openai

# Define your OpenAI API key
openai.api_key = "your_openai_api_key"

# Path to the image file and caption .txt file
image_path = "path/to/your/image.jpg"
text_path = image_path.replace('jpg','txt') 

# Open the image in binary mode
with open(image_path, "rb") as image_file:
    # Call the OpenAI API with the image file
    response = openai.ChatCompletion.create(
        model="gpt-4-vision",  # Ensure you have access to GPT-4 Vision
        messages=[
            {"role": "system", "content": "You are an AI that captions images."},
            {"role": "user", "content": "Please caption this image."}
        ],
        files={"file": image_file}  # Pass the image file directly
    )

# Extract and print the caption
caption = "JLSCOM " + response["choices"][0]["message"]["content"].strip()

# write the caption to .txt file
with open(image_path, "w") as text_file:
    text_file.write(caption)

```

The dateset should be formated in a specific way where the image and caption files have the same name and are in the same folder. 

``` Shell
dataset
 |-img0.jpg
 |-img0.txt
 |-img1.jpg
 |-img1.txt
 |- ...
```

# Training config 

To train the model using the scripts of `ai-toolkit` please copy  `train_lora_flux_schnell_24gb_JLSCOM.yaml` to `ai-toolkit/config`. 

```shell
cp train_lora_flux_schnell_24gb_JLSCOM.yaml ai-toolkit/config
```

The file represents all the configurations needed to train a `LoRA adapted` to fine-tune the `Flux1.0-Shnell` model. 

```yaml
job: extension # train a LoRA adapted 
config:
  # this name will be the folder and filename name
  name: "JLSCOM_garment_LoRA_flux_schnell"
  process:
    - type: 'sd_trainer' # The traininf script, AI-toolkit uses the SDTrainer scripts from https://github.com/kohya-ss/sd-scripts.git
      # root folder to save training sessions/samples/weights
      training_folder: "path/to/output/training/folder"
      device: cuda:0 # device to accelerate calculations
      trigger_word: "JLSCOM" # Trigger word 
      network:
        type: "lora" # type of fine-tuning (LoRA)
        linear: 32 # Rank
        linear_alpha: 32 # Rank
      save:
        dtype: float16 # precision level to save weights
        save_every: 50 # save every this many steps of updates during taining 
        max_step_saves_to_keep: 20 # how many intermittent saves to keep
        push_to_hub: true #change this to True to push your trained model to Hugging Face at the end of training.       
        hf_repo_id: Amine-CV/JLSCOM_garment_LoRA_flux_schnell_v1 #your Hugging Face repo ID to save the model
       
      datasets:
        # datasets are a folder of images. captions need to be txt files with the same name as the image
        # for instance image2.jpg and image2.txt. Only jpg, jpeg, and png are supported currently
        # images will automatically be resized and bucketed into the resolution specified
        # on windows, escape back slashes with another backslash so
        # "C:\\path\\to\\images\\folder"
        - folder_path: "../imgs"
          caption_ext: "txt"
          caption_dropout_rate: 0.05  # will drop out the caption 5% of time
          shuffle_tokens: false  # shuffle caption order, split by commas
          cache_latents_to_disk: true  # leave this true unless you know what you're doing
          resolution: [ 512, 768, 1024 ]  # flux enjoys multiple resolutions
      train:
        batch_size: 1
        steps: 4000  # total number of steps to train 500 - 4000 is a good range
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false  # probably won't work with flux
        gradient_checkpointing: true  # need the on unless you have a ton of vram
        noise_scheduler: "flowmatch" # for training only
        optimizer: "adamw8bit"
        lr: 5e-6
        # uncomment this to skip the pre training sample
        skip_first_sample: true
        # uncomment to completely disable sampling
#        disable_sampling: true
        # uncomment to use new bell curved weighting. Experimental but may produce better results
#        linear_timesteps: true

        # ema will smooth out learning, but could slow it down. Recommended to leave on.
        ema_config:
          use_ema: true
          ema_decay: 0.99

        # will probably need this if gpu supports it for flux, other dtypes may not work correctly
        dtype: bf16
      model:
        # huggingface model name or path
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter" # Required for flux schnell training
        is_flux: true
        quantize: true  # run 8bit mixed precision
        # low_vram is painfully slow to fuse in the adapter avoid it unless absolutely necessary
        low_vram: true  # uncomment this if the GPU is connected to your monitors. It will use less vram to quantize, but is slower.
      sample:
        sampler: "flowmatch" # must match train.noise_scheduler
        sample_every: 50 # sample every this many steps
        width: 1024
        height: 1024
        prompts:
          # you can add [trigger] to the prompts here and it will be replaced with the trigger word
#          - "[trigger] holding a sign that says 'I LOVE PROMPTS!'"\
          - "[trigger] Garment Type: Slim-Fit Jeans Fit and Style: Slim-fit, designed to hug the legs closely without being overly tight, offering a contemporary, streamlined appearance. Color and Pattern: Soft pastel green in a solid shade, adding a subtle pop of color to outfits while maintaining a minimalist, modern look. Fabric/Material: Crafted from a stretch cotton blend, providing comfort, flexibility, and durability. Details: Traditional five-pocket design with two front pockets, two back pockets, and a small coin pocket, all seamlessly integrated for functionality and style. Display Style: Displayed in a flat lay to highlight the overall structure and color. Background and Lighting: Set against a light gray background with soft, even lighting to bring out the pastel hue of the jeans without overshadowing it. Shape: Fitted shape with a tapered leg, maintaining a sleek and tailored silhouette from hip to ankle. Closures: Secured with a standard button and zipper fly in matching tones for a seamless look. Branding: Minimal branding with a discreet internal label; no external logos, maintaining a clean, understated aesthetic. Cuffs and Hems: Clean, stitched hems at the ankle, allowing the jeans to be worn full-length or slightly rolled for a casual look. Fit: Slim yet comfortable, allowing ease of movement while staying fitted through the legs. Length: Full length, designed to sit right at the ankle, suitable for pairing with both casual and semi-formal footwear. Occasion: Versatile enough for both casual daily wear and smart-casual occasions, adding a fresh twist to any wardrobe. Style Influence: Inspired by modern minimalist fashion, with a focus on clean lines and a refined color palette. Seasonal Suitability: Ideal for spring and summer wear due to the light color and breathable fabric. Texture: Smooth, soft finish with a hint of stretch, ensuring comfort during prolonged wear. Weight: Medium weight, suitable for warm weather without feeling too thin. Finish: Matte finish, enhancing the soft, pastel tone for a polished, sophisticated look. Aesthetic Style: Casual chic, blending comfort with a contemporary style that is effortlessly versatile. Target Audience: Suitable for individuals seeking stylish yet comfortable jeans with a unique color that is easy to style. Ease of Care: Machine washable, with colorfastness to retain the pastel shade after multiple washes."
          - "[trigger] Garment Type: Blazer Fit and Style: Regular fit with a tailored, classic style that combines formality with a modern touch. Color and Pattern: Soft sage green in a solid color, providing a subtle, sophisticated look. Fabric/Material: Made from a lightweight, smooth wool blend, offering both comfort and a structured appearance. Details: Features two-button closure at the front, with a notched lapel collar for a classic blazer look. Includes a single welt pocket on the chest and two large patch pockets at the lower front for a stylish yet practical detail. Display Style: Displayed in a flat lay to clearly showcase the entire structure and details of the blazer. Background and Lighting: Light gray background with soft, even lighting to emphasize the blazer's color and details without distractions. Shape: Structured, slightly fitted shape that follows a traditional blazer silhouette. Sleeve Style: Long sleeves with a slight taper towards the wrist, finished with simple cuffs. Closures: Front two-button closure with matte black buttons, adding to the minimalistic and versatile design. Branding: Discreet internal branding tag near the neckline, keeping the outer look clean and unbranded. Pockets: Three pockets total'one welt pocket on the chest and two patch pockets at the front, providing both functional and decorative elements. Cuffs and Hems: Standard blazer hemline with neatly finished cuffs. Fit: Tailored, providing a structured and polished look without being overly slim. Length: Standard blazer length, designed to sit at the hip. Occasion: Suitable for formal occasions, work, or smart-casual settings, adding a refined touch to any outfit. Style Influence: Minimalist with classic tailoring, suitable for a modern professional look. Seasonal Suitability: Ideal for spring and autumn due to its lightweight wool blend material. Texture: Smooth with a slight sheen, giving a refined and polished appearance. Weight: Medium weight, providing structure without being too heavy, ideal for layering over shirts or light sweaters. Finish: Matte finish, keeping the look understated and versatile for various settings. Aesthetic Style: Clean, classic, and modern, suitable for professionals and style-conscious individuals. Target Audience: Individuals looking for a versatile blazer that can transition from formal work settings to more relaxed social gatherings. Ease of Care: Dry clean recommended to maintain fabric quality and structure."
          - "[trigger] Garment Type: Denim Shorts Fit and Style: Regular fit with a classic, relaxed style, offering comfort without being overly tight or baggy. Color and Pattern: Light blue with a subtle faded wash, giving it a vintage-inspired look and adding visual texture to the fabric. Fabric/Material: Made from soft, breathable denim, likely a cotton blend for comfort and durability. Details: Traditional five-pocket design with two front pockets, two back pockets, and a small coin pocket within the right front pocket. Button closure and zipper fly for secure fastening. Display Style: Displayed in a flat lay, showcasing the shorts' full length, pockets, and seams. Background and Lighting: Light gray background with soft lighting, highlighting the light blue denim color and subtle fading. Shape: Straight shape with a relaxed fit, ensuring comfort around the thighs and hips. Closures: Front button and zipper closure, a classic feature in denim shorts for easy wear. Embellishments: Minimal design with no additional embellishments, maintaining a clean and simple aesthetic. Branding: Discreet brand tag on the internal waistband; no external logos. Patterns/Prints: Solid denim with a faded wash effect, adding dimension without any specific patterns or prints. Pockets: Five-pocket design for both style and functionality, allowing for small item storage. Cuffs and Hems: Standard hem style, ending just above the knee, offering a casual and versatile length. Fit: Regular fit that balances comfort and style, suitable for various body types. Length: Knee-length, ideal for warm weather and casual wear. Occasion: Perfect for casual, everyday wear during spring and summer, or for outdoor and relaxed settings. Style Influence: Classic and timeless, inspired by traditional denim styles with a slight vintage wash for a modern twist. Seasonal Suitability: Best suited for spring and summer, given its breathable fabric and knee-length design. Texture: Smooth denim texture, slightly faded for added character. Weight: Medium weight, providing durability without being too heavy, ideal for warmer climates. Finish: Matte finish with a faded wash effect, giving it a lived-in, relaxed appearance. Aesthetic Style: Simple, casual, and versatile, fitting well into a variety of everyday wardrobes. Target Audience: Suitable for individuals seeking a comfortable and classic pair of denim shorts for casual wear. Ease of Care: Machine washable for easy maintenance, durable enough for frequent wear."
          - "[trigger] Garment Type: Chino Pants Fit and Style: Regular fit with a classic, casual style, offering a comfortable and versatile silhouette. Color and Pattern: Solid olive green, adding a subtle yet stylish touch to a neutral wardrobe. Fabric/Material: Made from a durable cotton blend, providing comfort with a slight stretch for flexibility. Details: Includes belt loops and a button with zipper closure, adding functionality while maintaining a clean look. Display Style: Displayed in a flat lay, highlighting the shape and structure of the pants. Background and Lighting: Light gray background with soft, even lighting that enhances the color and minimalist design. Shape: Straight-leg shape with a relaxed but structured fit, ideal for casual and semi-formal occasions. Closures: Front button and zipper closure for ease of wear and a secure fit. Embellishments: No embellishments, keeping the design simple and versatile. Branding: Likely includes a discreet internal label; no visible external branding for a clean look. Patterns/Prints: Solid color with no patterns or prints, emphasizing a minimalist, versatile aesthetic. Pockets: Four-pocket design, featuring two front slash pockets and two rear welt pockets, providing practical storage without disrupting the streamlined appearance. Cuffs and Hems: Finished hems with a slight roll-up detail, adding a subtle casual touch. Fit: Regular fit, offering a balance between tailored and relaxed for everyday comfort. Length: Full length, intended to reach the ankle or slightly above when rolled. Occasion: Suitable for casual outings, smart-casual settings, or weekend wear, pairing well with a variety of tops. Style Influence: Minimalist and classic, with a utilitarian feel that makes it adaptable for different styles. Seasonal Suitability: Ideal for spring and fall, with a medium-weight fabric that offers comfort in mild weather. Texture: Smooth and slightly textured, providing a refined look with a soft touch. Weight: Medium weight, suitable for transitioning between seasons. Finish: Matte finish, enhancing the subtle elegance and versatility of the pants. Aesthetic Style: Classic and functional, appealing to those who prefer understated, adaptable wardrobe staples. Target Audience: Perfect for individuals seeking comfortable yet stylish pants that can be dressed up or down. Ease of Care: Machine washable for easy maintenance, making it a practical addition to any wardrobe."
          - "[trigger] Garment Type: Half-Zip Sweater Fit and Style: Regular fit with a slightly tailored silhouette, offering a balanced look that isn't too snug or too loose, suitable for layering. Color and Pattern: Dark heathered gray on the main body with solid black on the sleeves, shoulders, and cuffs. The sweater has a subtle color-blocked design that adds contrast and visual interest. Fabric/Material: Made from a soft and durable blend of cotton and polyester, providing warmth and breathability. Details: Features a quarter-length zip closure at the neckline for adjustability and styling flexibility, along with a single welt pocket on the left chest for added functionality and style. Display Style: Displayed in a flat lay style, giving a clear view of the design and structure of the garment. Background and Lighting: Set against a neutral gray background with soft, even lighting that highlights the contrast between the gray and black areas of the sweater. Shape: Slightly tapered at the waist and cuffs, providing a clean and structured look. Sleeve Style: Long sleeves in solid black, contributing to the color-blocked effect. Neckline: High collar with a half-zip for added versatility and a modern, sporty look. Closures: Half-zip closure at the neck with a metal zipper, allowing for easy adjustments to the neckline. Cuffs and Hems: Ribbed cuffs and hem to provide a snug fit and help retain warmth, adding to the overall polished look. Fit: Regular fit that maintains comfort and ease of movement, making it suitable for everyday wear. Length: Standard length, falling around the hip, suitable for both layering and wearing alone. Occasion: Ideal for casual outings, relaxed office environments, or cool-weather activities. Style Influence: A blend of sporty and casual influences, offering a versatile look that works well in urban or laid-back settings. Seasonal Suitability: Suited for fall and winter, with a design that provides warmth while being comfortable for transitional weather. Texture: Smooth fabric on the body with a slight heathered effect, contrasting with the solid, slightly textured black on the sleeves. Weight: Medium weight, providing warmth without being overly bulky. Finish: Matte finish, lending a sophisticated and modern look to the garment. Layer Style: Can be layered over a T-shirt or under a jacket for added warmth. Collar Type: High collar with zip functionality, providing both style and protection against cooler weather. Aesthetic Style: Modern, minimalist with a sporty edge, appealing to those who prefer subtle yet stylish details. Size Labeling: Standard fit, to accommodate various body types. Climate Suitability: Perfect for mild to cool climates where a lightweight yet warm layer is needed. Target Audience: Designed for individuals who enjoy practical, comfortable, and stylish apparel that can transition between casual and slightly dressed-up looks. Style Preferences: Appeals to those who prefer versatile, understated fashion with a hint of athletic influence. Ease of Care: Machine washable for easy maintenance and durability. This half-zip sweater combines practicality and style, with a modern color-blocked design that provides visual contrast. Its regular fit, warm fabric, and versatile look make it a reliable choice for cooler weather, perfect for layering or wearing on its own."
          - "[trigger] Garment Type: Chino Pants Fit and Style: Tailored slim-fit chinos, offering a modern and polished appearance suitable for both casual and semi-formal occasions. Color and Pattern: Soft beige color with a solid, clean finish, making it versatile and easy to pair with various tops. Fabric/Material: Made from a high-quality cotton blend, ensuring breathability and comfort with a slight stretch for ease of movement. Details: Features a classic design with belt loops, a front button closure, and a zip fly. There are no visible logos, maintaining a minimalist aesthetic. Display Style: Displayed in a flat lay format to showcase the structure and length clearly. Background and Lighting: Set against a light cream background with soft, diffused lighting to enhance the neutral tone of the fabric. Shape: Slim, tapered shape that offers a clean silhouette without appearing too tight. Closures: Front button and zipper closure provide a secure and classic look. Pockets: Includes two side pockets and two back welt pockets, adding functionality without disrupting the streamlined appearance. Cuffs and Hems: Finished with a standard hem at the ankles, giving a neat and classic look that works well with both dress shoes and casual footwear. Fit: Slim-fit, offering a modern look that's both stylish and comfortable. Length: Full-length chinos that reach the ankles, suitable for a variety of occasions. Occasion: Ideal for both casual and business-casual settings; can be dressed up or down with ease. Style Influence: Inspired by classic menswear with a touch of contemporary minimalism, making it a wardrobe staple. Seasonal Suitability: Suitable for all seasons, particularly spring and autumn, due to the breathable cotton material. Texture: Smooth and soft to the touch, providing a comfortable wear experience. Weight: Lightweight, making it comfortable for day-long wear. Finish: Matte finish, contributing to the sophisticated look of the pants. Aesthetic Style: Clean and minimal, catering to a classic yet contemporary style preference. Target Audience: Designed for individuals seeking a polished look that balances style with comfort. Ease of Care: Machine washable and easy to care for, adding to its practicality for everyday wear. These chinos are a versatile, refined addition to any wardrobe, offering a polished, understated style suitable for both work and leisure. Their neutral color and tailored fit make them ideal for a range of outfit combinations, providing an effortlessly stylish look."
          - "[trigger] Garment Type: Turtleneck sweater Fit and Style: Regular fit, slightly relaxed for a cozy yet polished appearance, suitable for layering or wearing alone. Color and Pattern: Solid, creamy off-white color with no additional patterns, giving it a clean, minimalist look. Fabric/Material: Made from a soft, warm wool or wool-blend fabric, ideal for maintaining warmth while providing breathability. Details: Ribbed texture around the turtleneck, cuffs, and hem for added style and durability, with a fine knit texture across the body and sleeves. Display Style: Displayed in a flat lay style, showing the entire sweater's design and structure clearly. Background and Lighting: Set against a light grey background with soft, diffused lighting, creating a neutral and inviting mood that complements the sweater's color. Shape: Straight shape with a slightly boxy silhouette, designed to drape comfortably without clinging to the body. Sleeve Style: Long sleeves with ribbed cuffs that provide a snug fit around the wrists, adding a refined detail to the overall design. Neckline: High turtleneck collar, ribbed for texture and warmth, designed to fold over or stand upright depending on the wearer's preference. Closures: No closures, emphasizing the pullover style of the sweater. Embellishments: None, keeping the look simple and classic. Branding: No visible branding, aligning with a minimalist, unbranded style. Cuffs and Hems: Ribbed cuffs and hem add subtle structure and a tailored finish to the overall relaxed fit. Fit: Regular fit, providing enough room for comfort while still maintaining a tailored look. Length: Standard length, ending around the hip, making it versatile for pairing with jeans, trousers, or skirts. Occasion: Suitable for casual, business casual, or semi-formal settings, depending on styling. Style Influence: Minimalist and classic, perfect for a timeless wardrobe. Seasonal Suitability: Ideal for fall and winter wear due to the insulating material and high neckline. Texture: Soft, smooth wool texture with a ribbed finish on the turtleneck, cuffs, and hem. Weight: Medium-weight fabric, balancing warmth and comfort. Finish: Matte finish, enhancing its cozy appearance. Aesthetic Style: Simple yet chic, suitable for those who appreciate a minimalist, high-quality look. Target Audience: Designed for individuals seeking versatile, comfortable, and elegant winter clothing. Ease of Care: Likely requires gentle washing or dry cleaning to maintain quality. This turtleneck sweater is a wardrobe staple that combines comfort, warmth, and a refined, minimalist style, perfect for a range of casual to semi-formal settings."
          - "[trigger] Garment Type: Corduroy Jacket Fit and Style: Relaxed fit with a minimalist, utilitarian style. Color and Pattern: Light cream, solid color with a smooth, uniform look. Fabric/Material: Made from corduroy, giving it a soft, textured feel with visible ridges. Details: Features two large, square patch pockets on the chest, adding a functional and balanced look. Display Style: Displayed in a flat lay, showcasing the full front of the jacket. Background and Lighting: Beige background with soft, diffused lighting, emphasizing the jacket's light color and texture. Shape: Slightly boxy shape, providing a relaxed, unstructured silhouette. Sleeve Style: Long sleeves with a standard cuff style, ending at the wrist. Neckline: Features a classic spread collar for a clean, polished look. Closures: Full-length front zipper closure, adding a modern, streamlined look. Embellishments: No additional embellishments, maintaining a minimalist aesthetic. Branding: Discreet inner label tag at the neckline; no external branding visible. Patterns/Prints: Solid color with a natural corduroy texture. Pockets: Two front chest pockets with a square shape and open top for easy access. Cuffs and Hems: Simple, finished hems on sleeves and bottom; no added detail for a clean look. Fit: Relaxed fit for comfortable layering over other clothing. Length: Standard jacket length, ending just above the hip. Occasion: Suitable for casual wear, offering a versatile layer for everyday outfits. Style Influence: Minimalist and utilitarian, inspired by workwear aesthetics. Seasonal Suitability: Ideal for fall and mild winter days due to its slightly heavier fabric. Texture: Soft corduroy texture with subtle ridges for a tactile, cozy feel. Weight: Medium weight, providing some warmth without being bulky. Finish: Matte finish, maintaining the fabric's natural appearance. Layer Style: Great as a top layer over T-shirts or light sweaters. Collar Type: Spread collar, giving a structured yet casual vibe. Aesthetic Style: Simple, functional aesthetic with a nod to vintage workwear. Target Audience: Perfect for individuals who value understated, practical clothing with a hint of vintage charm. Ease of Care: Likely machine washable, with care recommended for the corduroy fabric."
          - "[trigger] Garment Type: Quarter-Zip Knit Sweater Fit and Style: This sweater has a relaxed yet tailored fit, making it suitable for layering over shirts or wearing solo for a polished, casual look. Color and Pattern: Light heather grey, with a solid color and no additional patterns, offering a minimalist aesthetic. Fabric/Material: Crafted from a soft wool-blend knit, providing warmth and comfort while remaining lightweight. Details: Features a quarter-zip closure with a metal zipper, adding a modern touch and versatility to the classic sweater design. No visible logos or branding, keeping the look clean and sophisticated. Display Style: Displayed flat lay, showcasing the sweater's structure and fit without distractions. Background and Lighting: Set against a soft beige background with gentle lighting to highlight the texture and neutral tone of the sweater. Shape: Straight silhouette with a slightly fitted shape, ensuring a comfortable and refined appearance. Sleeve Style: Long sleeves with ribbed cuffs that fit snugly around the wrists, adding structure to the design. Neckline: Polo-style collar with a quarter-zip that allows for adjustable coverage at the neck, creating options for styling. Closures: Quarter-zip closure located at the center front, offering an adjustable neckline. Cuffs and Hems: Ribbed cuffs and hem to provide a structured fit and prevent stretching over time. Fit: Relaxed fit that maintains a tailored appearance, suitable for both casual and semi-formal settings. Length: Falls at the hip, offering a classic sweater length that pairs well with various bottoms. Occasion: Ideal for smart-casual and business-casual settings, as well as relaxed outings during cooler seasons. Style Influence: Minimalist design with a modern touch, perfect for understated, versatile wardrobes. Seasonal Suitability: Suitable for fall and winter due to the wool-blend material, which offers warmth without bulk. Texture: Soft knit texture that gives a cozy and refined feel, adding a subtle visual interest. Weight: Medium-weight, ensuring warmth and comfort without feeling overly heavy. Finish: Matte finish, contributing to the sweater's sophisticated and understated aesthetic. Layer Style: Works well as a top layer over shirts or under coats, providing versatility for different looks. Collar Type: Polo-style collar with a zipper, blending classic and contemporary elements. Aesthetic Style: Simple, clean, and chic, suitable for those who appreciate refined yet relaxed pieces. Target Audience: Designed for individuals seeking a blend of comfort, warmth, and style in cooler months. Ease of Care: Likely requires gentle washing, such as hand-washing or machine-washing on a wool cycle to maintain its texture and quality. This light heather grey quarter-zip sweater is a versatile and stylish addition, perfect for layering and providing warmth while maintaining a minimalist, modern appeal."
          - "[trigger] Garment Type: Drawstring Trousers Fit and Style: Relaxed-fit trousers with a casual yet polished aesthetic, suitable for both informal and semi-formal settings. Color and Pattern: Dark charcoal gray with a subtle, solid pattern that offers a sleek, versatile look. Fabric/Material: Likely a soft blend of cotton and polyester, providing comfort, durability, and a hint of stretch. Details: Features an elastic waistband with a drawstring closure for adjustable comfort, and two side pockets for functionality. Display Style: Displayed in a flat lay, allowing a clear view of the garment's shape, style, and details. Background and Lighting: Set against a light, neutral background with soft lighting, emphasizing the trousers' dark tone and clean lines. Shape: Straight-leg cut that gives a streamlined silhouette, with a slightly tapered look at the hem for a modern feel. Closures: Elasticated waistband with a drawstring, allowing for a secure, customizable fit without the need for a belt. Pockets: Two slanted side pockets for convenient storage, designed to be functional without disrupting the garment's smooth lines. Cuffs and Hems: Simple hem style, giving a neat finish to the trouser legs. Fit: Relaxed fit, balancing comfort with a tailored appearance. Length: Full-length trousers that fall straight to the ankles, versatile for various occasions. Occasion: Suitable for casual outings, work-from-home days, or even dressed up for a smart-casual event. Style Influence: Minimalist and modern, with a hint of athleisure influence due to the drawstring waistband. Seasonal Suitability: Ideal for year-round wear, thanks to its versatile color and comfortable material. Texture: Smooth, with a slight texture that adds depth to the dark color without detracting from the overall sleekness. Weight: Medium-weight fabric, suitable for layering in cooler weather or as standalone wear in moderate climates. Aesthetic Style: Casual chic with a functional design, bridging the gap between casual comfort and refined style. Target Audience: Designed for individuals seeking a comfortable yet stylish option for casual or semi-formal wear. Ease of Care: Likely machine washable, making it easy to care for and maintain. These dark charcoal drawstring trousers offer a versatile addition to any wardrobe, combining relaxed comfort with a polished, minimalist aesthetic. The elastic waistband and soft fabric make them ideal for all-day wear, while the streamlined silhouette allows for effortless styling across different occasions."
        neg: ""  # not used on flux
        seed: 42
        walk_seed: true
        guidance_scale: 1  # schnell does not do guidance
        sample_steps: 4  # 1 - 4 works well
# you can add any additional meta info here. [name] is replaced with config name at top
meta:
  name: "JLSCOM_garment_LoRA_flux_schnell"
  version: '1.0'
  description: "Fine-Tuning FLUX1-Schnell model using LoRA on Jules.com (JLSCOM) garment dataset. The dataset contains images of various garments with their descriptions. The trigger word is JLSCOM."
  license: "AGPL-3.0"
  

```


The name od the module : 

```yaml
        name: "JLSCOM_garment_LoRA_flux_schnell"

```

Tpo specify : 
1. The folder of the outputs where weights will be saved `training_folder`. 
2. The trigger word that we use to generate the specific style we trained on `trigger_word`.

```yaml
        training_folder: "path/to/output/training/folder"
        device: cuda:0 # device to accelerate calculations
        trigger_word: "JLSCOM" # Trigger word 
```
To specify the type of the Fine-tuning using `LoRA` (Low Rank Adapters). 
`linear` and `linear_alpha` specify the Rank 

```yaml
        network:
            type: "lora" # type of fine-tuning (LoRA)
            linear: 32 # Rank
            linear_alpha: 32 # Rank

```
To save the model weights please specify the : 
1. The `dtype` for the precision of the model; 
2. `save_evry`  to control the interval of saving checkpoints
3. `max_step_saves_to_keep` to control how many checkpoints should be saved
4. `push_to_hub` and `hf_repo_id` to be able to push the model automatically to the hugging-face hub. (Note : `HF_Token` should be defined in the `env`). 

```yaml
      save:
        dtype: float16 # precision level to save weights
        save_every: 50 # save every this many steps of updates during taining 
        max_step_saves_to_keep: 20 # how many intermittent saves to keep
        push_to_hub: true #change this to True to push your trained model to Hugging Face at the end of training.       
        hf_repo_id: Amine-CV/JLSCOM_garment_LoRA_flux_schnell_v1 #your Hugging Face repo ID to save the model

```
In the dataset configuration pleasr specify : 
 1. The `folder_path` to the dataset. 
 2. The `caption_ext` to the caption files type.

```yaml
      datasets:
          folder_path: "../imgs"
          caption_ext: "txt"
```
To specify : 
1.  `name_or_path` , `assistant_lora_path` specify the model and `LoRA` weights. 
2. `quantize` to quantize the model 
3.  `low_vram` offloading the model to CPU when not in use, reduces the use of VRAM, but slows down the training. 


```yaml
      model:
        # huggingface model name or path
        name_or_path: "black-forest-labs/FLUX.1-schnell"
        assistant_lora_path: "ostris/FLUX.1-schnell-training-adapter" # Required for flux schnell training
        is_flux: true
        quantize: true  # run 8bit mixed precision
        # low_vram is painfully slow to fuse in the adapter avoid it unless absolutely necessary
        low_vram: true  # uncomment this if the GPU is connected to your monitors. It will use less vram to quantize, but is slower.
```


```yaml

```


```yaml

```
