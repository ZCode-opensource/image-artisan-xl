## Why only Stable Diffusion XL

I’m a big fan of Stable Diffusion XL 1.0! Once I tried it, I couldn’t go back to using the old models. I know that most people still prefer the old ones since they have more models and require less power, but in my opinion, they don’t compare with XL when used with good models and LoRAs.

Image Artisan XL only supports this model architecture because of time restrictions. It would take too long to adapt the code to use all the other models out there. If there’s one in the future that competes with Stable Diffusion XL, I would probably make it compatible with this software.

## Prompt restriction

Note: This is primarily my personal perspective based on experimentation, understanding the models employed, and reviewing academic papers. However, my area of expertise lies outside the field of neural networks and their creation, so my opinion may be considered amateur.

Stable Diffusion XL exhibits a superior comprehension of prompt coherence compared to older versions. However, when employing techniques to bypass the CLIP token limit, the prompt degrades in quality, hindering the Unet's ability to interpret it effectively. In my view, this diminishes one of the significant advancements of Stable Diffusion XL.

Furthermore, enabling more tokens tends to incentivize individuals to copy and paste excessively lengthy prompts that incorporate repetitive words and phrases that don't significantly impact the quality of the generated image. This behavior stems from the tendency to utilize this model like its predecessors, employing a list of tags separated by commas (Danbooru style), which functions but falls short in facilitating detailed image composition

This prompt which has `45 tokens` and guides the composition (altough it doesn't work all the time):

> "breathtaking high-quality image of a big old tree with a dog at the right side with an airplane in the sky, golden hour, beautiful lake in the background, professional, high budget hollywood move production, cinematic, film grain"

It's not the same as this one where everything is left at random with danbooru style tags (`27 tokens`)

> "breathtaking, high-quality, big old tree, dog, dawn, professional, high budget hollywood move production, cinematic, film grain"

Compared to the last one, the following doesn't add any quality or composition to the final result, it just changes the image randomly (`109 tokens`).

> "breathtaking, high-quality, big old tree, dog, airplane, sky, golden hour, beautiful lake, background, professional, high budget hollywood move production, cinematic, film grain, soft dimmed light, cinematic photo, (cinematic still:1.2), bokeh, photograph, ultrarealistic, professional, 4k, highly detailed, detailed hands, perfect hands, ultra detailed skin, small skin imperfections, award winning masterpiece with incredible details, Non-representational, colors and shapes, expression of feelings, imaginative, highly detailed"

And almost all the images shared in sites are like the last one with an even bigger negative prompt which Stable Diffusion XL almost never needs.
