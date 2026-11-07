"""Head-ownership boundary checks; no model imports/execution."""
import unittest
from scripts.train_receiver_heads import head_groups,HEADS


class ReceiverHeadTests(unittest.TestCase):
    def test_frozen_and_shared_owners_are_excluded(self):
        proto,score,shared,d=object(),object(),object(),object()
        groups=[{'name':'D','lr':2e-5,'weight_decay':0.,'named_parameters':[('layer.D_A',d)]},
                {'name':'new_other','lr':2e-4,'weight_decay':0.,'named_parameters':[('codec.prototypes',proto),('codec.status.bias',shared)]},
                {'name':'score','lr':1e-4,'weight_decay':0.,'named_parameters':[('codec.score_row',score)]}]
        selected=head_groups(groups)
        self.assertEqual({n for g in selected for n,_ in g['named_parameters']},HEADS)
        self.assertEqual(selected[0]['lr'],2e-4);self.assertEqual(selected[1]['lr'],1e-4)
        self.assertIs(selected[0]['named_parameters'][0][1],proto)
        self.assertEqual(len(groups[1]['named_parameters']),2)

    def test_missing_or_duplicate_head_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'two-head'):
            head_groups([{'named_parameters':[('codec.score_row',object())]}])
        with self.assertRaisesRegex(ValueError,'two-head'):
            head_groups([{'named_parameters':[('codec.score_row',object()),('codec.score_row',object()),('codec.prototypes',object())]}])

if __name__=='__main__':unittest.main()
