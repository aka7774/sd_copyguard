import json
from modules import processing

last_infotext = None
copy_count = 1
copy_limit = 3

def process_images_inner_ex(p: processing.StableDiffusionProcessing) -> processing.Processed:
    global last_infotext, copy_count, copy_limit

    if p.all_prompts is None:
        p.all_prompts = [p.prompt]
    if p.all_negative_prompts is None:
        p.all_negative_prompts = [p.negative_prompt]
    if p.all_seeds is None:
        p.all_seeds = [p.seed]
    if p.all_subseeds is None:
        p.all_subseeds = [p.subseed]

    infotext = processing.create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds)

    # Count
    if infotext.find("Seed: -1") != -1:
        pass
    elif infotext == last_infotext:
        copy_count += 1
        print(f"{infotext} {copy_count}")
    else:
        last_infotext = infotext
        copy_count = 1

    # Overwrite Random
    if copy_count >= copy_limit:
        print(f"Copyguard!")
        p.seed = -1
        i = 0
        for seed in p.all_seeds:
            p.all_seeds[i] = -1
            i += 1

    return process_images_inner_original(p)

process_images_inner_original = processing.process_images_inner
processing.process_images_inner = process_images_inner_ex
