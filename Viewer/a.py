import cv2
import wx


class ImagePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        self.bitmap = wx.Bitmap("/Viewer/preview.jpg")
        #image = bitmap.ConvertToImage()
        #self.SetSize(parent.GetSize())
        self.SetSize((800, 600))
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        #self.Bind(wx.EVT_SIZING, self.OnPaint)
        #wx.StaticBitmap(self,-1, bitmap, (10,10), (980,550))
        #size = self.GetSize()  
        #image = image.Scale(size[0], size[1], wx)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()
        #self.Refresh()
        image = self.bitmap.ConvertToImage()
        panel_size = self.GetSize()
        image_size = self.bitmap.GetSize()
        size = self.CalcSize(image_size, panel_size)
        print(size, self.GetSize(),self.bitmap.GetSize())
        image = image.Rescale(size[0], size[1], wx.IMAGE_QUALITY_HIGH)
        #edge = cv2.Canny(image,100, 200)
        x = image.RGBValue()
        bmp = wx.Bitmap(image)
        dc.DrawBitmap(bmp,panel_size[0]/2-size[0]/2,panel_size[1]/2-size[1]/2)

    def CalcSize(self, image_size, panel_size):
        width_scale = panel_size[0] / image_size[0]
        height_scale = panel_size[1] / image_size[1]
        if width_scale*image_size[1] > panel_size[1]:
            return (int(height_scale*image_size[0]), panel_size[1])
        return (panel_size[0], int(width_scale*image_size[1]))
        

  

class Viewer(wx.Frame):
    
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "Previwer")
        self.SetSize((1200, 800))
        self.SetMinSize((500, 300))
        self.image_panel = ImagePanel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.image_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, 20)
        self.SetSizer(vbox)
        self.Bind(wx.EVT_SIZING, self.OnResize)
        self.Bind(wx.EVT_MAXIMIZE, self.OnResize)
        self.Show()

    def OnResize(self, event):
        self.image_panel.Refresh()

if __name__ == "__main__":
    app = wx.App(False)
    #frame = wx.Frame(None, wx.ID_ANY, "Preview")
    f = Viewer(None)
    a = cv2.imread("preview.jpg")
    b = cv2.Canny(a,100,200)
    f.Show(True)
    app.MainLoop()
    
