from enum import Enum


class FairnessMode(Enum):
    
    demo_parity = 'demographic parity'
    equality_odds = 'equality of odds'
    equality_oppr = 'equality of opportunity'

    def __str__(self):
        return self.value
