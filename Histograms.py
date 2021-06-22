import imageio
import matplotlib.pyplot as plt
import glob
import seaborn as sns
import numpy as np

#get all tif images
imPath = "path/to/images"
KMean_images = [imPath+"All_Data_6.tif", imPath+"CTX_NoD_DCS_6.tif", imPath+"NoD_6.tif"]
images = glob.glob(imPath + '*_8bit.tif')


for i in range(len(images)):
    for j in range(len(KMean_images)):
        image_name = images[i].replace(imPath,"")
        image_name = image_name.replace("_", " ").replace(".tif", "")
        kmean_image_name = KMean_images[j].replace(imPath,"")
        kmean_image_name = kmean_image_name.replace("_"," ").replace(".tif","")
        print(image_name)
        print(kmean_image_name)
    
        #load in the image
        img = imageio.imread(images[i])
        kimg = imageio.imread(KMean_images[j]) 
        print("loaded images")
        
        plt.close()
        print("cleared previous plot")
        
        #create plot features
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Density")
        plt.title("Pixel Intensity vs Density {} masked by {} K=6".format(image_name.replace(" georef 8bit",""), kmean_image_name.replace(" 6","")))
        print("set plot details")
        
        #set up plot classes
        cls0 = img[np.where(kimg==0)].flatten()
        cls1 = img[np.where(kimg==1)].flatten()
        cls2 = img[np.where(kimg==2)].flatten()
        cls3 = img[np.where(kimg==3)].flatten()
        cls4 = img[np.where(kimg==4)].flatten()
        cls5 = img[np.where(kimg==5)].flatten()
        
        sns.kdeplot(data=[cls0, cls1, cls2, cls3, cls4, cls5], fill=False, common_norm=True, palette="rocket", linewidth=.9)
        print("finished plot")
        
        #save the final histogram
        savePath = "path/to/save/folder"
        plt.savefig(savePath+"KDE_"+image_name.replace(" georef 8bit","")+"vs "+kmean_image_name+".png")
        print("plot saved")
        print("\n")


print("done")









