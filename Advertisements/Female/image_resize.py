from PIL import Image 
import os
def main(): 
    try: 
         #Relative Path
        a=os.getcwd()+"\Teenagers/"
        c = os.listdir(a)
        
        print(c)
        
        for i in c:
            img = Image.open(a+i)  
            img = img.resize((640, 380)) 
            img.save(a+"resized_image"+i)  
    except IOError: 
        pass
  
if __name__ == "__main__": 
    main() 
