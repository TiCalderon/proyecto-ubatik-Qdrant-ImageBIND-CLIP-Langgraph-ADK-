import fitz
doc = fitz.open("data/pdf/arch2.pdf")
page = doc[2] # page 3
print("Page size:", page.rect)
for img in page.get_images(full=True):
    print("Image:", img)
