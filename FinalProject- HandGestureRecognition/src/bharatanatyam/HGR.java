package bharatanatyam;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.swing.*;

public class HGR {
	private static final int kernelSize = 5;
	private static final double stdDev = 1.5;

	public static void main(String[] args) {
		JFrame webcamFrame = new JFrame(); //main GUI frame
		webcamFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		webcamFrame.setSize(1000, 1000);
		JPanel titlePanel = new JPanel(); //title panel
		titlePanel.setLayout(new FlowLayout(FlowLayout.CENTER));
		ImagePanel captureImagePanel = new ImagePanel(); //webcam image panel
		ImagePanel binaryPanel = new ImagePanel(); //binary image conversion panel
		ImagePanel contourPanel = new ImagePanel(); //contours drawn panel
		ImagePanel recognitionPanel = new ImagePanel(); //gesture recognized image conversion panel
		JPanel imagePanel = new JPanel(new BorderLayout());
		JPanel imageContourPanel = new JPanel(new BorderLayout());

		JLabel projectTitle = new JLabel("Hand Gesture Recognition Software for Bharatanatyam");
		titlePanel.add(projectTitle);

		webcamFrame.setLayout(new BorderLayout());
		webcamFrame.add(titlePanel, BorderLayout.PAGE_START);
		captureImagePanel.setPreferredSize(new Dimension(400, 400));
		imagePanel.add(captureImagePanel, BorderLayout.WEST);
		binaryPanel.setPreferredSize(new Dimension(400, 400));
		imagePanel.add(binaryPanel, BorderLayout.EAST);
		webcamFrame.add(imagePanel, BorderLayout.CENTER);
		contourPanel.setPreferredSize(new Dimension(400, 400));
		imageContourPanel.add(contourPanel, BorderLayout.WEST);
		recognitionPanel.setPreferredSize(new Dimension(400, 400));
		imageContourPanel.add(recognitionPanel, BorderLayout.EAST);
		webcamFrame.add(imageContourPanel, BorderLayout.PAGE_END);


		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		//Video Capture from webcam
		VideoCapture cap = new VideoCapture(0);
		cap.open(0);

		Mat capturedFrame = new Mat();
		Mat grayFrame = new Mat();
		Mat blurFrame = new Mat();
		Mat thresholdFrame = new Mat();

		//check if webcam is open
		if(!cap.isOpened()) {
			System.out.println("No Webcam Found");
		} else 
			System.out.println("Webcam Found" + cap.toString());
		webcamFrame.setVisible(true);

		if (cap.isOpened()) {
			try {
				Thread.sleep(500);
				while (true) {
					cap.read(capturedFrame);
					if (!capturedFrame.empty()) {
						try {
							Thread.sleep(200);

							//show webcam video in panel
							captureImagePanel.setImage(captureImagePanel.matToBufferedImage(capturedFrame));
							captureImagePanel.repaint();

							//background subtraction, extracting ROI
							Imgproc.cvtColor(capturedFrame, grayFrame , Imgproc.COLOR_RGB2GRAY);
							Imgproc.GaussianBlur(grayFrame, blurFrame, new Size(kernelSize,kernelSize), stdDev);
							Imgproc.threshold(blurFrame, thresholdFrame, 70, 255, Imgproc.THRESH_BINARY_INV+Imgproc.THRESH_OTSU);

							//finding contours
							Mat newImgT = thresholdFrame.clone();
							List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
							Mat hierarchy = new Mat();
							Imgproc.findContours(newImgT, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

							//extracting largest contour
							double maxArea = 1000;
							int contourIdx = -1;
							for(int i=0; i<contours.size(); ++i) {
								double area = Imgproc.contourArea(contours.get(i));
								if (area>maxArea) {
									maxArea=area;
									contourIdx = i;
								}
							}
							MatOfPoint tempContours = new MatOfPoint(contours.get(contourIdx));
							MatOfInt hull = new MatOfInt();
							MatOfInt4 defects = new MatOfInt4();

							//displaying largest contour and convex hull
							//Mat drawing = Mat.zeros(thresholdFrame.size(), capturedFrame.type());
							for(int i=0; i<contours.size(); ++i) {
								Imgproc.drawContours(capturedFrame, new ArrayList<MatOfPoint>(Arrays.asList(contours.get(i))), 0, new Scalar(0, 255, 0), 3);
							}

							//drawing convex hull
							//finding convexity defects
							/* Convexity defects is start_index
								end_index
								farthest_pt_index
								fixpt_depth*/
							Imgproc.convexHull(tempContours, hull, false);

							int contourIndex[] = hull.toArray();
							MatOfPoint contourPoints = new MatOfPoint();
							contourPoints.alloc(contourIndex.length);
							for(int j=0;j<contourIndex.length;j++) {
								contourPoints.put(j, 0, tempContours.get(contourIndex[j], 0)[0], tempContours.get(contourIndex[j], 0)[1]);
							}
							Imgproc.drawContours(capturedFrame, new ArrayList<MatOfPoint>(Arrays.asList(contourPoints)), 0, new Scalar(0, 0, 255), 3);

							Imgproc.convexityDefects(tempContours, hull, defects);
							List<Integer> cdList = defects.toList();
							MatOfPoint cdPoints = new MatOfPoint();
							cdPoints.alloc(cdList.size()/4);
							for(int j=2;j<cdList.size();j=j+4) {
								cdPoints.put(j/4, 0, tempContours.get(cdList.get(j), 0)[0], tempContours.get(cdList.get(j), 0)[1]);
							}
							
							Imgproc.drawContours(capturedFrame, new ArrayList<MatOfPoint>(Arrays.asList(cdPoints)), 0, new Scalar(255, 0, 0), 3);

							//displaying binary hand shape with contours on to the panel
							binaryPanel.setImage(binaryPanel.matToBufferedImage(thresholdFrame));
							binaryPanel.repaint();
							contourPanel.setImage(contourPanel.matToBufferedImage(capturedFrame));
							contourPanel.repaint();

						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
				}
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		cap.release();
	}
}

