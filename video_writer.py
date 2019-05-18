import time
import os
import cv2

def images_2_gif(image_folder, video_name='video', fps=25, loop=1, reverse=True):
    """
    Convert sequence of images to gif
    """
    import moviepy.editor as mpy
    
    #get variables
    video_name = video_name + "_" + str(round(time.time())) + ".gif"

    #Get images
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)

    # join paths
    for k in range(len(images)):
        images[k] = os.path.join(image_folder, images[k])

    gif = mpy.ImageSequenceClip(images, fps=fps)
    gif.write_gif(os.path.join(image_folder, video_name), fps=fps)

    return True

def images_2_video(image_folder, video_name='video', fps=25, loop=1, reverse=True):
    """
    Converts a set of images into a video
    """    
    #get variables
    video_name = video_name + "_" + str(round(time.time())) + ".avi" 
    
    reverse=True
    if reverse <= 0:
        reverse = False

    #Get images
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images = sorted(images)

    # get images' dimensions (assuming same)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # set video writer
    video = cv2.VideoWriter(os.path.join(image_folder, video_name), 0, fps, (width, height))

    #Generate video
    for l in range(loop):
        if loop > 1 and l > 1 and reverse:
            images = images[::-1]
        for k in range(len(images)):
            video.write(cv2.imread(os.path.join(image_folder, images[k])))

    return True