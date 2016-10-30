from PIL import Image
import sys

def processImage(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print "Cant load", infile
        sys.exit(1)
    i = 0
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            save_img_name = infile.split('.')[0]+'.jpg'
            print(save_img_name)
            new_im.save(save_img_name)

            i += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass # end of sequence
with open('files_list','r') as fread:
    for line in fread.readlines():
        img_file = line.split(' ')[0]
        print(img_file)
        if img_file.endswith('gif'):
            processImage(img_file)

