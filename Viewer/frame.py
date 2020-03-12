import wx
import numpy as np
import cv2
import matplotlib as plt
import functions as f


class ImageHolders(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        self.original_image = ImagePanel(self)
        self.modified_image = ModifiedImagePanel(self)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.original_image, 1, wx.EXPAND | wx.ALL, 20)
        hbox.Add(self.modified_image, 1, wx.EXPAND | wx.ALL, 20)
        self.SetSizer(hbox)

    def UpdateFile(self,file_name):
        self.original_image.UpdateFile(file_name)
        self.modified_image.UpdateFile(file_name)
    
    def UpdateMethod(self, method):
        self.modified_image.UpdateMethod(method)

class ImagePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        try:
            self.bitmap = wx.Bitmap()
        except:
            self.bitmap = wx.Bitmap(1, 1, 1).Create()

        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def UpdateFile(self,file_name):
        self.bitmap = wx.Bitmap(file_name)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        dc.Clear()
        image = self.bitmap.ConvertToImage()
        panel_size = self.GetSize()
        image_size = self.bitmap.GetSize()
        
        size = self.CalcSize(image_size, panel_size)
        image = image.Rescale(size[0], size[1], wx.IMAGE_QUALITY_HIGH)
        bmp = wx.Bitmap(image)
        dc.DrawBitmap(bmp, int(panel_size[0]/2-size[0]/2), int(panel_size[1]/2-size[1]/2))

    def CalcSize(self, image_size, panel_size):
        width_scale = panel_size[0] / image_size[0]
        height_scale = panel_size[1] / image_size[1]
        if width_scale*image_size[1] > panel_size[1]:
            return (int(height_scale*image_size[0]), panel_size[1])
        return (panel_size[0], int(width_scale*image_size[1]))
   

class ModifiedImagePanel(ImagePanel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        try:
            self.bitmap = wx.Bitmap()
        except:
            self.bitmap = wx.Bitmap(1, 1, 1).Create()
            
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def UpdateFile(self,file_name):
        self.bitmap = wx.Bitmap(file_name)
        self.original = self.bitmap

    def UpdateMethod(self, method):
        self.bitmap = method(self.original)


class Viewer(wx.Frame):
    
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, "Previwer")
        self.SetSize((1200, 800))
        self.SetMinSize((500, 300))

        self.image_panel = ImageHolders(self)

        self.file_dialog = wx.FileDialog(self, "Open", "", "", "", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        self.file_select = wx.Button(self, wx.ID_ANY, "Choose image")

        method_names = ["Histogram", "Canny", "Gaussian blur"]
        self.methods = {"Histogram":  f.Histogram, "Canny" : f.Canny, "Gaussian blur": f.GaussianBlur}
        self.choose_method = wx.Choice(self, choices=method_names)

        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.file_select, 0,0, 20)
        vbox.Add(self.choose_method, 0,0, 20)
        vbox.Add(self.image_panel, 1, wx.EXPAND | wx.ALL, 20)
        self.SetSizer(vbox)

        self.file_select.Bind(wx.EVT_BUTTON, self.OnPress)
        self.Bind(wx.EVT_SIZING, self.OnResize)
        self.Bind(wx.EVT_MAXIMIZE, self.OnResize)
        self.choose_method.Bind(wx.EVT_CHOICE, self.OnChoice)

        self.Show()

    def OnPress(self, event):
        self.file_dialog.ShowModal()
        self.image_panel.UpdateFile(self.file_dialog.GetPath())
        x = self.choose_method.GetSelection()
        if x != -1:
            self.image_panel.UpdateMethod(self.methods[self.choose_method.GetString(self.choose_method.GetSelection())])
        self.image_panel.Refresh()

    def OnResize(self, event):
        self.image_panel.Refresh()

    def OnChoice(self, event):
        
        self.image_panel.UpdateMethod(self.methods[self.choose_method.GetString(self.choose_method.GetSelection())])
        self.image_panel.Refresh()

if __name__ == "__main__":
    
    app = wx.App(False)
    f = Viewer(None)
    f.Show(True)
    app.MainLoop()
