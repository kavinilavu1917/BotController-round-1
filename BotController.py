import math
import cv2
import time
from threading import Thread
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
from socket import *
import warnings
import colorama
from colorama import Fore, Style
warnings.filterwarnings("ignore")
colorama.init()

class InitiativePointCalculation():
	def __init__(self):
		cap = cv2.VideoCapture(0)
		while 1:
			ret, frame = cap.read()

			self.image=frame
			if ret:
				self.start_point, self.turning_point, self.dest_point=self.Constrain_bot(frame,'Bot1')
				BotFunctionJSON['Bot1']['start_point']=self.start_point
				BotFunctionJSON['Bot1']['turning_point']=self.turning_point
				BotFunctionJSON['Bot1']['dest_point']=self.dest_point
				cv2.line(self.image,self.start_point, self.turning_point,(255,0,0),2)
				cv2.line(self.image,self.turning_point, self.dest_point,(255,0,0),2)

				

				self.start_point, self.turning_point, self.dest_point=self.Constrain_bot(frame,'Bot2')
				BotFunctionJSON['Bot2']['start_point']=self.start_point
				BotFunctionJSON['Bot2']['turning_point']=self.turning_point
				BotFunctionJSON['Bot2']['dest_point']=self.dest_point
				cv2.line(self.image,self.start_point, self.turning_point,(0,255,0),2)
				cv2.line(self.image,self.turning_point, self.dest_point,(0,255,0),2)

				self.start_point, self.turning_point, self.dest_point=self.Constrain_bot(frame,'Bot3')
				BotFunctionJSON['Bot3']['start_point']=self.start_point
				BotFunctionJSON['Bot3']['turning_point']=self.turning_point
				BotFunctionJSON['Bot3']['dest_point']=self.dest_point
				cv2.line(self.image,self.start_point, self.turning_point,(0,0,255),2)
				cv2.line(self.image,self.turning_point, self.dest_point,(0,0,255),2)


				
				self.start_point, self.turning_point, self.dest_point=self.Constrain_bot(frame,'Bot4')
				BotFunctionJSON['Bot4']['start_point']=self.start_point
				BotFunctionJSON['Bot4']['turning_point']=self.turning_point
				BotFunctionJSON['Bot4']['dest_point']=self.dest_point
				cv2.line(self.image,self.start_point, self.turning_point,(255,0,0),2)
				cv2.line(self.image,self.turning_point, self.dest_point,(255,0,0),2)
				

				break
		cap.release()


	def calculate_points(self,image,color_code,x1,y1):
		lower_red = np.array(color_code[0])
		upper_red = np.array(color_code[1])
		hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
		mask = cv2.inRange(hsv,lower_red,upper_red)
		res = cv2.bitwise_and(image,image, mask=mask)
		gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
		edged = cv2.Canny(gray,1,220)
		thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY)[1]
		coords = np.column_stack(np.where(thresh.transpose()>0))
		rotrect = cv2.minAreaRect(coords)
		x,y=list(map(int,rotrect[0]))
		x=x+x1
		y=y+y1
		return (x,y)

	def Constrain_bot(self,image,bot_name):
		height, width,_ = image.shape
		a,b = divmod(height,2)
		y0,y1,y2 = 0,a,a+a+b
		a,b = divmod(width,3)
		x0,x1,x2,x3 = 0,a,a+a,a+a+a+b
		if bot_name in ['Bot1','Bot2']:
			points = [
				self.calculate_points(image[y0:y1, x1:x2],BotFunctionJSON[bot_name]['track'],x1,y0),
				self.calculate_points(image[y1:y2, x1:x2],BotFunctionJSON[bot_name]['track'],x1,y1),
				self.calculate_points(image[y1:y2, x0:x1],BotFunctionJSON[bot_name]['track'],x0,y1)
			]

		else:
			points = [
				self.calculate_points(image[y0:y1, x1:x2],BotFunctionJSON[bot_name]['track'],x1,y0),
				self.calculate_points(image[y1:y2, x1:x2],BotFunctionJSON[bot_name]['track'],x1,y1),
				self.calculate_points(image[y1:y2, x2:x3],BotFunctionJSON[bot_name]['track'],x2,y1)
			]
		return points

class ImageProcessing():
	def __init__(self,color_code,start_point, dest_point,p_point1, p_point2,verticle,forward):
		print (Fore.LIGHTYELLOW_EX)
		print (start_point,dest_point,p_point1,p_point2,color_code,verticle,forward)
		self.a_path = start_point[1] - dest_point[1]
		self.b_path = start_point[0] - dest_point[0]
		self.start_point = start_point
		self.dest_point  = dest_point
		self.verticle=verticle
		self.forward=forward
		self.lowercode = color_code[0]
		self.uppercode = color_code[1]
		self.p_line = [
						p_point1[1] - p_point2[1],
						p_point1[0] - p_point2[0],
						((p_point1[0] - p_point2[0])*dest_point[1]) - ((p_point1[1] - p_point2[1])*dest_point[0])
					]
		print (self.p_line)
		print (Style.RESET_ALL)

	def ImagePrediction(self,image):
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		lower = np.array(self.lowercode)
		upper = np.array(self.uppercode)
		mask = cv2.inRange(hsv, lower,upper)
		res = cv2.bitwise_and(image,image, mask=mask)
		gray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray,(5,5),0)
		edged = cv2.Canny(gray,100,220)
		kernel = np.ones((5,5),np.uint8)
		closed=cv2.morphologyEx(edged,cv2.MORPH_CLOSE,kernel)
		thresh = cv2.threshold(edged, 0, 255, cv2.THRESH_BINARY)[1]
		coords = np.column_stack(np.where(thresh.transpose()>0))
		rotrect = cv2.minAreaRect(coords)
		box = cv2.boxPoints(rotrect)
		box = np.int0(box)
		length1 = ((box[0,0]-box[1,0])**2 + (box[0,1]-box[1,1])**2)**0.5
		length2 = ((box[1,0]-box[2,0])**2 + (box[1,1]-box[2,1])**2)**0.5
		if length1<length2:
			x1 = (box[0][0] + box[1][0]) // 2
			y1 = (box[0][1] + box[1][1]) // 2
			x2 = (box[2][0] + box[3][0]) // 2
			y2 = (box[2][1] + box[3][1]) // 2
		else:
			x1 = (box[0][0] + box[3][0]) // 2
			y1 = (box[0][1] + box[3][1]) // 2
			x2 = (box[1][0] + box[2][0]) // 2
			y2 = (box[1][1] + box[2][1]) // 2
		

		self.mid_point = [int(i) for i in rotrect[0]]

		self.c_path = self.b_path*self.mid_point[1] - self.a_path*self.mid_point[0]
		self.static_path_line = [ self.a_path, self.b_path, self.c_path]
		self.dynamic_path_line = [
			self.mid_point[1] - self.dest_point[1],
			self.mid_point[0] - self.dest_point[0],
			(self.mid_point[0] - self.dest_point[0])*self.mid_point[1] - (self.mid_point[1] - self.dest_point[1])*self.mid_point[0]
		]

		if self.verticle:
			if y1>y2:
				self.top_point=(x1,y1)
				self.bottom_point = (x2,y2)
			else:
				self.top_point=(x2,y2)
				self.bottom_point=(x1,y1)
		else:
			if x1<x2:
				self.top_point=(x1,y1)
				self.bottom_point=(x2,y2)
			else:
				self.top_point=(x2,y2)
				self.bottom_point=(x1,y1)
		if not self.forward:
			self.top_point, self.bottom_point = self.bottom_point, self.top_point

		self.distance = float('%.2f' %(((self.mid_point[0]-self.dest_point[0])**2 + (self.mid_point[1]-self.dest_point[1])**2)**0.5))

		try:
			self.path_slop = float('%.2f' %((self.mid_point[1]-self.dest_point[1])/(self.mid_point[0]-self.dest_point[0])))
		except ZeroDivisionError:
			self.path_slop = math.inf 

		self.path_angle=math.atan(self.path_slop)*180/(math.pi)
		if self.path_angle<0:
			self.path_angle = 180+self.path_angle

		try:
			self.static_path_slop = float('%.2f' %(-1*self.a_path/self.b_path))
		except ZeroDivisionError:
			self.static_path_slop = math.inf

		self.static_path_angle = math.atan(self.static_path_slop)*180/math.pi
		if self.static_path_angle<0:
			self.static_path_angle = 180+self.static_path_angle


		try:
			self.bot_slop = float('%.2f' %((self.top_point[1]-self.bottom_point[1])/(self.top_point[0]-self.bottom_point[0])))
		except ZeroDivisionError:
			self.bot_slop = math.inf

		self.bot_angle=math.atan(self.bot_slop)*180/(math.pi)
		if self.bot_angle<0:
			self.bot_angle = 180+self.bot_angle

		self.dynamic_angle=float('%.2f' %(abs(self.path_angle-self.bot_angle)))
		if self.dynamic_angle>90:
			self.dynamic_angle = 180-self.dynamic_angle
		self.dynamic_angle = float('%.2f' %self.dynamic_angle)	

		self.static_angle = float('%.2f' %(abs(self.static_path_angle-self.bot_angle)))
		if self.static_angle>90:
			self.static_angle = 180-self.static_angle
		self.static_angle = float('%.2f' %self.static_angle)


		self.dynamic_position = self.find_position_using_two_point_and_dest_point(self.dynamic_path_line, self.top_point)
		self.static_position  = self.find_position_using_two_point_and_dest_point(self.static_path_line, self.top_point)
		self.p_position       = self.find_position_using_two_point_and_dest_point(self.p_line, self.mid_point)

		cv2.line(image,self.mid_point,self.dest_point,(0,255,0),2)
		cv2.line(image,self.top_point, self.bottom_point, (255,0,0),2)
		cv2.line(image,self.top_point,self.dest_point,(0,255,246),2)
		cv2.drawContours(image, [box], 0, (0,0,255), 2)
		cv2.putText(image,'DAnlge:{}'.format(self.dynamic_angle),(10,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
		cv2.putText(image,'SAngle:{}'.format(self.static_angle), (10,110),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
		cv2.putText(image,'Distance:{}'.format(self.distance),(10,140),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
		cv2.putText(image,'Dpostion:{}'.format(self.dynamic_position),(10,170),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
		cv2.putText(image,'Sposition:{}'.format(self.static_position),(10,200),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
		cv2.putText(image,'Pposition:{}'.format(self.p_position),(10,230),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
		self.image=image

	def find_distance_between_two_points(self,point1,point2):
		distance = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
		return float('%.2f' %distance)

	def find_position_using_two_point_and_dest_point(self,line,Predict_point):
		if line[0]<0:
			line = [-1*i for i in line]
		res = line[0]*Predict_point[0] - line[1]*Predict_point[1] + line[2]

		if line[2]<0:
			if res<0:
				return -1
			elif res>0:
				return 1
			else:
				return 0
		elif line[2]>0:
			if res>0:
				return -1
			elif res<0:
				return 1
			else:
				return 0
		else:
			return 0



class BotMovement():
	def __init__(self,conn_obj,color_code, start_point, turning_point,dest_point,location):
		self.conn_obj     = conn_obj
		self.last_command = ''
		self.cap          = cv2.VideoCapture(0)
		if location=='left':
			self.feature=True 
			self.image_obj    = ImageProcessing(color_code,start_point, turning_point,turning_point,dest_point,True,True)
			print (Fore.RED+'[+] ForwardTrack!'+Style.RESET_ALL)
			self.ForwardTrack()
			print (Fore.RED+'[+] LeftTurn!'+Style.RESET_ALL)
			self.LeftTurn()
			self.image_obj    = ImageProcessing(color_code, turning_point, dest_point,start_point,turning_point,False,True)
			self.feature=False
			print (Fore.RED+'[+] ForwardTrack!'+Style.RESET_ALL)
			self.ForwardTrack()
#			self.MidTurn()
			self.conn_obj.send(b'N')
			self.feature=True
			self.image_obj    = ImageProcessing(color_code,dest_point, turning_point,start_point,turning_point,False,False)
			print (Fore.RED+'[+] BackwardTrack!'+Style.RESET_ALL)
			self.BackwardTrack()
			print (Fore.RED+'[+] RightTurn!'+Style.RESET_ALL)
			self.RightTurn()
			self.feature=False
			self.image_obj    = ImageProcessing(color_code,turning_point,start_point,turning_point,dest_point,True,False)
			print (Fore.RED+'[+] BackwardTrack!'+Style.RESET_ALL)
			self.BackwardTrack()
#			self.MidTurn()
		else:
			self.feature=True
			self.image_obj    = ImageProcessing(color_code,start_point, turning_point,turning_point,dest_point, True,True)
			print (Fore.RED+'[+] ForwardTrack!'+Style.RESET_ALL)
			self.ForwardTrack()
			print (Fore.RED+'[+] RightTurn!'+Style.RESET_ALL)
			self.RightTurn()
			self.image_obj    = ImageProcessing(color_code,turning_point,dest_point,start_point,turning_point,False,False)
			print (Fore.RED+'[+] ForwardTrack!'+Style.RESET_ALL)
			self.ForwardTrack()
			self.conn_obj.send(b'N')
			self.feature=False
			self.image_obj    = ImageProcessing(color_code,dest_point,turning_point,start_point,turning_point,False,True)
			print (Fore.RED+'[+] BackwardTrack!'+Style.RESET_ALL)
			self.BackwardTrack()
			print (Fore.RED+'[+] LeftTurn'+Style.RESET_ALL)
			self.LeftTurn()
			self.image_obj    = ImageProcessing(color_code,turning_point,start_point,turning_point,dest_point,True,False)
			print (Fore.RED+'[+] BackwardTrack!'+Style.RESET_ALL)
			self.BackwardTrack()
		self.conn_obj.close()
		self.cap.release()
		cv2.destroyAllWindows()

	def ForwardTrack(self):
		while 1:
			ret, frame = self.cap.read()
			if ret:
				self.image_obj.ImagePrediction(frame)
				cv2.imshow('Image', self.image_obj.image)
				if cv2.waitKey(1)&0xff==ord('q'):
					break

				if (self.feature and self.image_obj.p_position>0) or (not self.feature and self.image_obj.p_position<0):
					print (self.feature, self.image_obj.p_position)
					self.conn_obj.send(b'S')
					print (Fore.BLUE+'=> S command sent!'+Style.RESET_ALL)
					print(Fore.LIGHTYELLOW_EX+'[+] placed by PLine features!'+Style.RESET_ALL)
					self.last_command=''
					break

				if round(self.image_obj.distance)<=12:
					self.conn_obj.send(b'S')
					print (Fore.BLUE+'=> S command sent!'+Style.RESET_ALL)
					print (Fore.LIGHTYELLOW_EX+'[+] placed!'+Style.RESET_ALL)
					self.last_command=''
					break
				elif round(self.image_obj.distance)>12 and round(self.image_obj.dynamic_angle)<=10:
					self.conn_obj.send(b'S')
					time.sleep(0)
					self.conn_obj.send(b'F')
#					print ('=> F command sent!')
					self.last_command = 'F'
				else:
					if self.image_obj.dynamic_position<0 and self.last_command!='R':
						self.conn_obj.send(b'S')
						time.sleep(0)
						self.conn_obj.send(b'R')
						self.last_command='R'
						print (Fore.BLUE+'=> R command sent!'+Style.RESET_ALL)
					elif self.image_obj.dynamic_position>0 and self.last_command!='L':
						self.conn_obj.send(b'S')
						time.sleep(0)
						self.conn_obj.send(b'L')
						self.last_command='L'
						print (Fore.BLUE+'=> L command sent!'+Style.RESET_ALL)
			else:
				self.cap = cv2.VideoCapture(0)

	def MidTurn(self):
		while 1:
			ret, frame = self.cap.read()
			if ret:
				self.image_obj.ImagePrediction(frame)
				cv2.imshow('Image',self.image_obj.image)
				if cv2.waitKey(1)&0xff==ord('q'):
					break

				if round(self.image_obj.static_angle)<=10:
					self.conn_obj.send(b'S')
					self.last_command=''
					break
				else:
					if self.image_obj.static_position<0 and self.last_command!='R':
						self.conn_obj.send(b'S')
						time.sleep(0)
						self.conn_obj.send(b'R')
						self.last_command='R'
					elif self.image_obj.static_position>1 and self.last_command!='L':
						self.conn_obj.send(b'S')
						time.sleep(0)
						self.conn_obj.send(b'L')
						self.last_command='L'
			else:
				self.cap = cv2.VideoCapture(0)

	def LeftTurn(self):
		while 1:
			ret, frame = self.cap.read()
			if ret:
				self.image_obj.ImagePrediction(frame)
				cv2.imshow('Image',self.image_obj.image)
				if cv2.waitKey(1)&0xff==ord('q'):
					break
				if self.image_obj.static_position<0 and self.image_obj.static_angle>10:
					self.conn_obj.send(b'S')
					self.last_command=''
					print (Fore.LIGHTYELLOW_EX+'=> S command sent!'+Style.RESET_ALL)
					break

				if self.image_obj.static_position>0 and self.last_command!='L':
					self.conn_obj.send(b'S')
					time.sleep(0)
					self.conn_obj.send(b'L')
					self.last_command='L'
					print (Fore.BLUE+'=> L command sent!'+Style.RESET_ALL)
			else:
				self.cap = cv2.VideoCapture(0)

	def BackwardTrack(self):
		while 1:
			ret, frame = self.cap.read()
			if ret:
				self.image_obj.ImagePrediction(frame)
				cv2.imshow('Image', self.image_obj.image)
				if cv2.waitKey(1)&0xff==ord('q'):
					break
				if (self.feature and self.image_obj.p_position>0) or (not self.feature and self.image_obj.p_position<0):
					print (self.feature, self.image_obj.p_position)
					self.conn_obj.send(b'S')
					print (Fore.BLUE+'=> S command sent!'+Style.RESET_ALL)
					print (Fore.LIGHTYELLOW_EX+'[+] placed by extra features!'+Style.RESET_ALL)
					self.last_command=''
					break

				if round(self.image_obj.distance)<=12:
					self.conn_obj.send(b'S')
					print (Fore.BLUE+'=> S command sent!'+Style.RESET_ALL)
					print (Fore.LIGHTYELLOW_EX+'[+] placed!'+Style.RESET_ALL)
					self.last_command=''
					break
				elif round(self.image_obj.distance)>12 and round(self.image_obj.dynamic_angle)<=10:
					self.conn_obj.send(b'S')
					time.sleep(0)
					self.conn_obj.send(b'B')
#					print ('=> B command sent!')
					self.last_command = 'B'
				else:
					if self.image_obj.dynamic_position<0 and self.last_command!='L':
						self.conn_obj.send(b'S')
						time.sleep(0)
						self.conn_obj.send(b'L')
						print (Fore.BLUE+'=> L command sent!'+Style.RESET_ALL)
						self.last_command='L'
					elif self.image_obj.dynamic_position>0 and self.last_command!='R':
						self.conn_obj.send(b'S')
						time.sleep(0)
						self.conn_obj.send(b'R')
						print (Fore.BLUE+'=> R command sent!'+Style.RESET_ALL)
						self.last_command='R'
			else:
				self.cap = cv2.VideoCapture(0)

	def RightTurn(self):
		while 1:
			ret, frame = self.cap.read()
			if ret:
				self.image_obj.ImagePrediction(frame)
				cv2.imshow('Image',self.image_obj.image)
				if cv2.waitKey(1)&0xff==ord('q'):
					break
				if self.image_obj.static_position<0 and self.image_obj.static_angle>10:
					self.conn_obj.send(b'S')
					print (Fore.LIGHTYELLOW_EX+'=> S command sent!'+Style.RESET_ALL)
					self.last_command=''
					break

				if self.image_obj.static_position>0 and self.last_command!='R':
					self.conn_obj.send(b'S')
					time.sleep(0)
					self.conn_obj.send(b'R')
					print (Fore.BLUE+'=> R command sent!'+Style.RESET_ALL)
					self.last_command='R'
			else:
				self.cap = cv2.VideoCapture(0)


def main():
	number_of_bot = 4

	initial_obj = InitiativePointCalculation()
	plt.imshow(initial_obj.image)
	plt.show()

	if input(Fore.LIGHTBLUE_EX+'[+] Can i start server y/n : '+Style.RESET_ALL)=='n':
		exit()

	host = 'localhost' #change host
	port = 12345

	sock_obj = socket(AF_INET,SOCK_STREAM)
	sock_obj.setsockopt(SOL_SOCKET, SO_REUSEADDR ,1)
	sock_obj.bind((host,port))
	sock_obj.listen(3)
	print (Fore.LIGHTGREEN_EX+'[+] Server Start {}:{}'.format(host,port)+Style.RESET_ALL)
	for i in range(number_of_bot):
		conn_obj, addr = sock_obj.accept()
		print (Fore.LIGHTYELLOW_EX+'{}'.format(addr)+Style.RESET_ALL)
		bot_name=conn_obj.recv(1024).decode()
		BotFunctionJSON[bot_name]['connObj']=conn_obj

	if input(Fore.LIGHTBLUE_EX+'[+] Detect y/n: '+Style.RESET_ALL)=='n':
		sock_obj.close()
		exit()
	print (Fore.LIGHTGREEN_EX+'[+] Bot 1 Movement start!'+Style.RESET_ALL)
	Bot1_obj = BotMovement(BotFunctionJSON['Bot1']['connObj'],BotFunctionJSON['Bot1']['color'],BotFunctionJSON['Bot1']['start_point'], BotFunctionJSON['Bot1']['turning_point'], BotFunctionJSON['Bot1']['dest_point'],'left')
	print (Fore.LIGHTGREEN_EX+'[+] Bot 2 Movement start!'+Style.RESET_ALL)
	Bot2_obj = BotMovement(BotFunctionJSON['Bot2']['connObj'],BotFunctionJSON['Bot2']['color'],BotFunctionJSON['Bot2']['start_point'], BotFunctionJSON['Bot2']['turning_point'], BotFunctionJSON['Bot2']['dest_point'],'left')
	print (Fore.LIGHTGREEN_EX+'[+] Bot 3 Movement start!'+Style.RESET_ALL)
	Bot3_obj = BotMovement(BotFunctionJSON['Bot3']['connObj'],BotFunctionJSON['Bot3']['color'],BotFunctionJSON['Bot3']['start_point'], BotFunctionJSON['Bot3']['turning_point'], BotFunctionJSON['Bot3']['dest_point'],'right')
	print (Fore.LIGHTGREEN_EX+'[+] Bot 4 Movement start!'+Style.RESET_ALL)
	Bot4_obj = BotMovement(BotFunctionJSON['Bot4']['connObj'],BotFunctionJSON['Bot4']['color'],BotFunctionJSON['Bot4']['start_point'], BotFunctionJSON['Bot4']['turning_point'], BotFunctionJSON['Bot4']['dest_point'],'right')

	sock_obj.close()


BotFunctionJSON = {
	'Bot1':{
		'track':[[168,196,146],[173,255,200]],
		'color':[[127,138,168],[149,182,193]],
		'location':'left'
	},
	'Bot2':{
		'track':[[64,146,49],[95,255,208]],
		'color':[[0,95,217],[0,172,255]],
		'location':'left'
	},
	'Bot3':{
		'track':[[19,101,142],[29,255,255]],
		'color':[[75,169,211],[131,255,255]],
		'location':'right'
	},
	'Bot4':{
		'track':[[144,134,193],[167,255,255]],
		'color':[[10,163,230],[31,236,255]],
		'location':'right'
	}
}
if __name__=='__main__':
	main()
