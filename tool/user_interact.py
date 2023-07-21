import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

def generate_trimap(rgb_image, foreground_mask, background_mask, unknown_threshold=30, output_file=None):
        # Convert the RGB image to grayscale
        grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        h, w = grayscale_image.shape

        # Create the initial trimap with all pixels marked as unknown (128)
        trimap = 128 * np.ones_like(grayscale_image, dtype=np.uint8)

        # Mark the foreground region as 255 and the background region as 0
        foreground_mask = foreground_mask.T
        background_mask = background_mask.T
        trimap[foreground_mask] = 255
        trimap[background_mask] = 0

        print("grayscale_image ", grayscale_image.shape)
        print("rgb_image ", rgb_image.shape)
        print("foreground_mask ", foreground_mask.reshape(h,w,1).repeat(3,2).mean(axis=-1).shape)
        print("background_mask ", background_mask.shape)
        # Compute the absolute differences between grayscale image and foreground and background
        fg_diff = np.abs(grayscale_image - (rgb_image*foreground_mask.reshape(h,w,1).repeat(3,2)).mean(axis=-1))#.sum(axis=-1)
        bg_diff = np.abs(grayscale_image - (rgb_image*background_mask.reshape(h,w,1).repeat(3,2)).mean(axis=-1))#.sum(axis=-1)
        print("fg_diff ", (grayscale_image - (rgb_image*foreground_mask.reshape(h,w,1).repeat(3,2)).mean(axis=-1)).sum(axis=1).shape)

        # Identify unknown regions based on the unknown threshold
        unknown_region = (fg_diff > unknown_threshold) & (bg_diff > unknown_threshold)
        # trimap[unknown_region] = 128

        if output_file:
            trimap_image = Image.fromarray(trimap)
            trimap_image.save(output_file)

class TrimapPainter:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trimap Painter")

        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack()

        self.foreground_mask = None
        self.background_mask = None

        self.outer_fg_mask = None
        self.outer_bg_mask = None
        self.inner_fg_mask = None
        self.inner_bg_mask = None

        self.draw_mode = False

        self.image = None  # Store the loaded image as a class attribute
        self.mask_size = 3

        self.image_path = None
        self.load_image_button = tk.Button(self.root, text="Load Image", command=self.load_image)
        self.load_image_button.pack()

        self.save_outer_circle = tk.Button(self.root, text="Save Unknown region", command=lambda: self.save_Trimap(True))
        self.save_outer_circle.pack()
        self.save_outer_circle["state"] = tk.DISABLED  # Disable the button initially

        self.save_inner_circle = tk.Button(self.root, text="Save Foreground region", command=lambda: self.save_Trimap(False))
        self.save_inner_circle.pack()
        self.save_inner_circle["state"] = tk.DISABLED  # Disable the button initially

        self.clear_button = tk.Button(self.root, text="Clear Drawing", command=self.clear_drawing)
        self.clear_button.pack()
        self.clear_button["state"] = tk.DISABLED  # Disable the button initially

        self.drawn_objects = []  # Store IDs of drawn objects (rectangles) on the canvas

        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.root.mainloop()

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.image = Image.open(self.image_path)
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image.width, height=self.image.height)  # Set canvas size to image size
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk)

            # Convert the RGB image to grayscale
            self.image = np.array(self.image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

            self.outer_fg_mask = 0 * np.ones_like(self.image, dtype=np.uint8)
            self.outer_bg_mask = 0 * np.ones_like(self.image, dtype=np.uint8)
            self.inner_fg_mask = 0 * np.ones_like(self.image, dtype=np.uint8)
            self.inner_bg_mask = 0 * np.ones_like(self.image, dtype=np.uint8)

            print(self.inner_fg_mask.shape)

    def clear_drawing(self):
        print("CLEARRRRR")
        for (obj_id, x, y) in self.drawn_objects:
            self.canvas.delete(obj_id)  # Delete the drawn objects (red rectangles) from the canvas
        self.foreground_mask = None
        self.background_mask = None
        self.drawn_objects = []  # Reset the list of drawn objects
        self.save_outer_circle["state"] = tk.DISABLED  # Disable the "Generate Trimap" button
        self.save_inner_circle["state"] = tk.DISABLED
        self.clear_button["state"] = tk.DISABLED  # Disable the "Clear Drawing" button

    def start_drawing(self, event):
        self.clear_drawing()
        self.draw_mode = True
        
    def update_masks(self, x, y):
        h, w = self.image.shape
        if self.foreground_mask is None or self.background_mask is None:
            print("SIZE:",h,w)
            self.foreground_mask = np.full((h, w), False)
            self.background_mask = np.full((h, w), True)

        x1, y1, x2, y2 = x, y, x + self.mask_size, y + self.mask_size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        for i in range(y1, y2):
            for j in range(x1, x2):
                try:
                    self.foreground_mask[i][j] = True
                    self.background_mask[i][j] = False
                except:
                    # print("ERROR: ", i, j)
                    # print("\t", x1, x2, y1, y2)
                    pass
        # # Bresenham's circle drawing algorithm
        # radius = self.mask_size // 2
        # cx, cy = x1 + radius, y1 + radius
        # for dx in range(-radius, radius + 1):
        #     for dy in range(-radius, radius + 1):
        #         if dx**2 + dy**2 <= radius**2:
        #             nx, ny = cx + dx, cy + dy
        #             self.foreground_mask[ny][nx] = True
        #             self.background_mask[ny][nx] = False

    def draw(self, event):
        if self.draw_mode:
            x, y = event.x, event.y
            rectangle_id = self.canvas.create_oval(x, y, x + self.mask_size, y + self.mask_size, fill="red")
            self.drawn_objects.append((rectangle_id, x, y))  # Store the ID of the drawn rectangle
            self.update_masks(x, y)

    def save_Trimap(self, isOuter):
        if isOuter:
            self.outer_fg_mask = self.foreground_mask
            self.outer_bg_mask = self.background_mask
        else:
            self.inner_fg_mask = self.foreground_mask
            self.inner_bg_mask = self.background_mask

        # Create the initial trimap with all pixels marked as bg (0)
        trimap = 0 * np.ones_like(self.image, dtype=np.uint8)

        # Mark the foreground region as 255 and the background region as 0
        foreground_mask = self.inner_fg_mask
        unknown_foreground_mask = self.outer_fg_mask
        
        trimap[unknown_foreground_mask] = 128
        trimap[foreground_mask] = 255

        output_file="generated_trimap.png"
        if output_file:
            trimap_image = Image.fromarray(trimap)
            trimap_image.save(output_file)

    def stop_drawing(self, event):
        self.draw_mode = False
        self.save_outer_circle["state"] = tk.NORMAL  # Enable the button after finishing drawing
        self.save_inner_circle["state"] = tk.NORMAL
        self.clear_button["state"] = tk.NORMAL

        mask = np.zeros_like(self.foreground_mask, dtype=np.uint8)
        for id, x, y in self.drawn_objects:
            cv2.circle(mask, (x, y), radius=3, color=255, thickness=-1)

        # Draw a polygon using the drawn_objects points
        points = [(x, y) for id, x, y in self.drawn_objects]
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # Mark pixels inside the filled area as foreground
        print(self.foreground_mask)
        self.foreground_mask[mask == 255] = True
        self.background_mask[mask == 255] = False

    def get_foreground_mask(self):
        return self.foreground_mask

    def get_background_mask(self):
        return self.background_mask
    
    def generate_trimap(self):
        if self.image_path and not self.foreground_mask is None:
            # Convert the RGB image to NumPy array
            image = Image.open(self.image_path)
            rgb_image = np.array(image)

            # Generate and save the trimap
            trimap = generate_trimap(rgb_image, self.foreground_mask, self.background_mask, output_file="generated_trimap.png")
            print("Trimap generated and saved as 'generated_trimap.png'.")

if __name__ == "__main__":
    painter = TrimapPainter()
    foreground_mask = painter.get_foreground_mask()
    background_mask = painter.get_background_mask()

    # You can use the generated foreground_mask and background_mask in the Poisson Matting algorithm.
