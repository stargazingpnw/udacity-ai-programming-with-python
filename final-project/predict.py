def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    # Resize & Center Crop
    img = img.resize((256, 256))
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    img = img.crop((left, top, left + 224, top + 224))
    
    # Normalize
    np_image = np.array(img) / 255
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(np_image.transpose((2, 0, 1))).float()

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    img = process_image(image_path).unsqueeze_(0).to(device)
    
    with torch.no_grad():
        logps = model.forward(img)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        
    # Move to CPU and convert to numpy for easy mapping
    top_p = top_p.cpu().numpy()[0]
    top_class = top_class.cpu().numpy()[0]
    
    # Invert the class_to_idx dictionary to get idx_to_class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    
    # Map the predicted indices to the actual class labels (folder names)
    classes = [idx_to_class[c] for c in top_class]
    
    return top_p, classes

image_path = "flowers/test/1/image_06743.jpg" # Example test image
top_k = 5

# 1. Run prediction
probs, classes = predict(image_path, model, top_k)

# 2. Map class labels (e.g., '1') to real names (e.g., 'pink primrose') 
# using the cat_to_name.json loaded earlier
flower_names = [cat_to_name[c] for c in classes]

# 3. Print results (Rubric Requirement)
print(f"\nResults for image: {image_path}")
print("-" * 30)
for i in range(len(flower_names)):
    print(f"Rank {i+1}: {flower_names[i]:<20} | Probability: {probs[i]:.4f}")
    
# TODO: Display an image along with the top 5 classes
image_path = "flowers/test/1/image_06743.jpg" # Example
probs, classes = predict(image_path, model)
names = [cat_to_name[c] for c in classes]

plt.figure(figsize=(5,10))
ax = plt.subplot(2,1,1)
imshow(process_image(image_path), ax=ax, title=names[0])

plt.subplot(2,1,2)
plt.barh(names, probs)
plt.gca().invert_yaxis()
plt.show()