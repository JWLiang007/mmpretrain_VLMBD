from mmpretrain import inference_model

result = inference_model('minigpt-4_vicuna-7b_caption', ['demo/cat-dog.png','demo/dog.jpg'],batch_size=2)
print(result)