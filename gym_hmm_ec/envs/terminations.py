 

class termination_base():

    def __init__(self,params) -> None:
        self.params = params
    def step(self):
        raise NotImplementedError
    def reset(self):
        raise NotImplementedError

class min_imitation_threshold(termination_base):

    def step(self,input_dict):
        if input_dict['motion_imitation'] < self.params['threshold']:
            return True
        else:
            return False
    def reset(self):
        pass


class mocap_epi_len(termination_base):

    def step(self,input_dict):
        self._n_step += 1
        if self._n_step >= input_dict['mocap_len']:
            return True
        else:
            return False
    def reset(self):
        self._n_step = 0