package bharatanatyam;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import javax.swing.JPanel;
import org.opencv.core.Mat;

public class ImagePanel extends JPanel {
    /**
     * 
     */
    private static final long serialVersionUID = -7065581626827263107L;
    private BufferedImage image; 

    public BufferedImage getImage(){  
        return image;  
    }  
    public void setImage(BufferedImage newImage){  
        image=newImage;  
    } 

    public BufferedImage matToBufferedImage(Mat matrix){
        int cols = matrix.cols();  
        int rows = matrix.rows();  
        int elemSize = (int)matrix.elemSize();  
        byte[] data = new byte[cols * rows * elemSize];  
        int type;  
        matrix.get(0, 0, data);  

        switch (matrix.channels()) {  
        case 1:  
            type = BufferedImage.TYPE_BYTE_GRAY;  
            break;  
        case 3:  
            type = BufferedImage.TYPE_3BYTE_BGR;  
            //BGR to RGB 
            byte b;  

            //Swap blue and red bytes
            for(int i=0; i<data.length; i=i+3) {  
                b = data[i];  
                  data[i] = data[i+2];  
                  data[i+2] = b;  
            }  
            break;  
        default:  
            return null;  
        }  

        //Create the new buffered image from the byte array data
        BufferedImage image2 = new BufferedImage(cols, rows, type);  
        image2.getRaster().setDataElements(0, 0, cols, rows, data);  
        return image2;  
    }

    @Override  
    protected void paintComponent(Graphics g){  
       super.paintComponent(g);   
       BufferedImage temp = this.getImage();  

       //Draw the buffered image
       if(temp != null)
           g.drawImage(temp,0,0,480,480, this);  //temp.getWidth() temp.getHeight()
    }  
}
