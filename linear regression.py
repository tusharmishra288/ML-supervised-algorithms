import numpy as np
import matplotlib.pyplot as plt


def estimatecoefficients(x,y):
    n=np.size(x)
    m_x,m_y=np.mean(x),np.mean(y)
    SS_xy=np.sum(y*x-n*m_x*m_y)
    SS_xx=np.sum(x*x-n*m_x*m_x)
    b1=SS_xy/SS_xx
    b0=m_y-b1*m_x
    return(b0,b1)
    
def plot_regressionline(x,y,b):
    plt.scatter(x,y,color='g',marker='o')
    y_pred=b[0]+b[1]*x
    plt.plot(x,y_pred,color='r')
    plt.xlabel('X-AXIS')
    plt.ylabel('Y-AXIS')
    plt.show()
    
def main():
     x=np.array([1,2,3,4,5,6,7,8,9,10])
     y=np.array([300,350,500,700,800,850,900,900,1000,1200])
     b=estimatecoefficients(x,y)
     print(b)
     plot_regressionline(x,y,b)

  
if __name__ == "__main__": 
    main()      
     
