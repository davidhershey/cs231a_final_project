
import numpy as np

"""
Extract features from a grayscale pong frame
"""
class PongExtractor(object):

    def __init__(self):
        self.oppLastY = None
        self.playerLastY = None
        self.ballLastX = None
        self.ballLastY = None
        self.OPP_X = 8
        self.PLAYER_X = 70
        self.paddle_range = (130,190)
        self.ball_range = (200,255)
        self.feature_length = 8

    def getPaddle(self,xcoord,frame):
        paddleCol =  np.squeeze(frame[:,xcoord])
        first = None
        last = None
        # print paddleCol
        for i in range(paddleCol.shape[0]):
            pixel = paddleCol[i]
            # print pixel
            if pixel > self.paddle_range[0] and pixel < self.paddle_range[1]:
                if first is None:
                    first = i
                last = i
        if first is None or last is None:
            return None
        return float(first+last)/2.0

    def getBall(self,frame):
        topLeft = None
        found = False
        for x in range(frame.shape[0]):
            if found:
                break
            for y in range(frame.shape[1]):
                pixel = frame[x,y]
                if pixel > self.ball_range[0] and pixel < self.ball_range[1]:
                    found = True
                    topLeft = (x,y)
                    break
        if topLeft is None:
            return None,None

        subImg = frame[x:x+7,y:y+7]
        botRight = None
        found = False
        xsize = subImg.shape[0]
        ysize = subImg.shape[1]
        for x in range(xsize-1,-1,-1):
            if found:
                break
            for y in range(ysize-1,-1,-1):
                pixel = subImg[x,y]
                if pixel > self.ball_range[0] and pixel < self.ball_range[1]:
                    botRight = (x+topLeft[0],y+topLeft[1])
                    found = True
                    break
        if botRight is None:
            botRight = topLeft
            # print frame
        xcoord = float(topLeft[0]+botRight[0])/2.0
        ycoord = float(topLeft[1]+botRight[1])/2.0
        return xcoord,ycoord



    def extract(self,frame):
        oppY = self.getPaddle(self.OPP_X,frame)
        playerY = self.getPaddle(self.PLAYER_X,frame)
        ballX,ballY = self.getBall(frame)

        # Oppenent derivative and update
        if oppY is None or self.oppLastY is None:
            dOpp = 0.0
        else:
            dOpp = oppY - self.oppLastY
        self.oppLastY = oppY
        if oppY is None:
            oppY = 40.0

        # Player derivative and update
        if playerY is None or self.playerLastY is None:
            dPlayer = 0.0
        else:
            dPlayer = playerY - self.playerLastY
        self.playerLastY = playerY
        if playerY is None:
            playerY = 40.0

        # Ball X derivative and update
        if ballX is None or self.ballLastX is None:
            dBx = 0.0
        else:
            dBx = ballX - self.ballLastX
        self.ballLastX = ballX
        if ballX is None:
            ballX = 40.0

        # Ball Y derivative and update
        if ballY is None or self.ballLastY is None:
            dBy = 0.0
        else:
            dBy = ballY - self.ballLastY
        self.ballLastY = ballY
        if ballY is None:
            ballY = 40.0

        features = [oppY,dOpp,playerY,dPlayer,ballX,ballY,dBx,dBy]
        return np.array(features)
