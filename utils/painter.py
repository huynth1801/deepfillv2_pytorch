import cv2

class Sketcher():
    def __init__(self, window_name, dests, colors, thick, type):
        self.prev_pt = None
        self.window_name = window_name
        self.dests = dests 
        self.colors = colors
        self.dirty = False
        self.thick = thick 

        self.show()

        if type == 'bbox':
            cv2.setMouseCallback(self.window_name, self.on_bbox)
        else:
            cv2.setMouseCallback(self.window_name, self.on_mouse)

    def small_thick(self):
        self.thick = max(3, self.thick - 1)

    def large_thick(self):
        self.thick = min(50, self.thick + 1)

    def show(self):
        cv2.imshow(self.window_name, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        loc = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_loc = loc
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_loc = None

        if self.prev_loc and flags and cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_loc, loc, color, self.thick)
            self.dirty = True
            self.prev_loc = loc
            self.show()     


    def on_bbox(self, event, x, y, flags, param):
        loc = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_loc = loc
        elif event == cv2.EVENT_LBUTTONUP:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.rectangle(dst, self.prev_loc, loc, color, -1)
            self.dirty = True
            self.prev_loc = None
            self.show()