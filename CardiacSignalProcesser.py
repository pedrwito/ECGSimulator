import numpy as np
import scipy.signal as signal


class Fruit:
    def some_func(some_text):
        print(some_text)


class Fruitas:
    @staticmethod
    def algo_func(algo_texto):
        print(algo_texto)



Fruit.some_func("hola")
f = Fruitas()
Fruitas.algo_func("hola")
f.algo_func("hola")


class CardiacSignalProcesser:

    @staticmethod
    def median_filter(csl, signal_, fs):

        med200 = signal.medfilt(np.array(signal_), [int(fs/5 + 1)])
        med600 = signal.medfilt(np.array(med200), [int(3*fs/5 + 1)])

        return np.subtract(signal_, med600)
    
    @staticmethod
    def correct_signal(signal_, fs, check_polarity = False):
        templates = CardiacSignalProcesser.__get_qrs_templates__(signal_, fs)
        
        if check_polarity:
            signal_, templates = CardiacSignalProcesser.__check_polarity__(signal_, templates)
            
        signal_ = CardiacSignalProcesser.__normalize_amplitude__(signal_, templates)
        return signal_
    
    @staticmethod        
    def __normalize_amplitude__(signal_, templates):
        templates_max = np.max(np.median(templates, axis=0))
        return signal_ / templates_max
    
    @staticmethod
    def __get_qrs_templates__(signal_, fs):
        i_peaks = CardiacSignalProcesser.R_peaks(signal_, fs)
        i_before = int(0.2*fs)
        i_after = int(0.4*fs)
        templates = []
        for i in i_peaks:
            template = signal_[(i-i_before):(i+i_after)]
            if len(template) == int(0.6*fs):
                templates.append(template)
        return np.array(templates)
    
    
    