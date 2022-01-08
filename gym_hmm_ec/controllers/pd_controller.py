import numpy as np

class PDController():
    
    def __init__(self,kps,kds,torque_max = None) -> None:
        
        self._kps = kps
        self._kds = kds

        if not torque_max == None:
            self.torque_max = torque_max
        else:
            self.torque_max = np.inf

    def get_torque(self,q_des,dq_des,q,dq):

        # print(np.degrees(q[7:9]), q_des[7:9], dq[7:9], dq_des[7:9])
        # print(np.degrees(q[13:15]), q_des[13:15], dq[13:15], dq_des[13:15])
        torque = self._kps*np.subtract(q_des,q) + self._kds*np.subtract(dq_des,dq)
        
        torque = np.clip(torque,-self.torque_max,self.torque_max)



        return torque
