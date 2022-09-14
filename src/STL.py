PRED_VARS = 'xy'

class STL:
    def __init__(self, stl_expr):
        # O(len(stl_expr))? About O(1)

        self.stl_expr = stl_expr
        self.parsed_tup = self.parse_stl(stl_expr)
        # self.hrz = self.gen_hrz()

    def get_parsed(self):
        return self.parsed_tup

    # def get_hrz(self, phi = None):
    #     return self.hrz

    # def gen_hrz(self, phi = None):
    #     if phi == None:
    #         phi = self.parsed_tup
        
    #     idfr = phi[0]
    #     if idfr in 'FG':
    #         b = 

    def rdegree(self, sig, phi = None, t = 0):
        if phi == None:
            phi = self.parsed_tup

        idfr = phi[0]
        if idfr in 'FG':
            op,(a,b),inner_phi = phi

            if len(sig) < t+b:
                raise Exception('Signal too short')

            t_prime = list(range(t+a,t+b+1))
            inner_rdegs = [self.rdegree(sig, inner_phi, tp) for tp in t_prime]

            if op == 'F':
                return max(inner_rdegs)
            elif op == 'G':
                return min(inner_rdegs)

        elif idfr in '&|':
            phis = phi[1:]
            inner_rdegs = [self.rdegree(sig,p,t) for p in phis]

            if idfr == '|':
                return max(inner_rdegs)
            if idfr == '&':
                return min(inner_rdegs)

        elif idfr in PRED_VARS:
            var,op,d = phi
            s = sig[t][var]
            if op == '<':
                return d-s
            elif op == '>':
                return s-d
            else:
                raise Exception('Unknown predicate comparator: {}'.format(op))
            
        else:
            raise Exception('Unknown identifier: {}'.format(idfr))

    def parse_stl(self, phi):
        # Not clear what the time complexity is, but not significant
        # O(len(phi))?
        if phi[0] in 'FG':
            temporal_op = phi[0]
            end = phi.index(']')
            comma = phi.index(',')
            a = int(phi[2:comma])
            b = int(phi[comma+1:end])
            inner_phi = phi[end+1:]
            info = self.parse_stl(inner_phi)
            return (temporal_op, (a,b), info)

        elif phi[0] == '(':
            tup = self.parse_and_or(phi)
            if len(tup) == 1:
                # Go again
                inner = self.parse_stl(tup[0])
                return inner

            expr1 = tup[0]
            op1 = tup[1]
            expr2 = tup[2]
            for op in tup[3::2]:
                if op != op1:
                    raise Exception("Ambiguous use of boolean operators without parentheses: {}".format(phi))
            # all same op
            exprs = [expr1,expr2]
            for e in tup[4::2]:
                exprs.append(e)

            infos = [self.parse_stl(e) for e in exprs]
            infos.insert(0,op1)
            return tuple(infos)
        elif phi[0] in PRED_VARS:
            # must be a predicate
            phi = phi.replace(' ','')
            var = phi[0]
            op = phi[1]
            num = float(phi[2:])
            return (var, op, num)
        else:
            raise Exception('Unparseable expression: {}'.format(phi))

    def parse_and_or(self, phi):
        """
        Returns a tuple of expressions seperated by an 'and' (&) or 'or' (|) 
        """
        # O(len(phi))

        parts = []
        depth = 0
        start = 0
        for i,c in enumerate(phi):
            if c == '(':
                depth += 1
                if depth == 1:
                    opr = phi[start:i].replace(' ','')
                    if opr not in '&|':
                        raise Exception("Invalid operator: {}".format(opr))
                    parts.append(opr)
                    start = i+1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    expr = phi[start:i]
                    parts.append(expr)
                    start = i+1
                elif depth < 0:
                    raise Exception("Mismatched parentheses in STL expression!")
        # Remove empty string at the front
        parts.pop(0)
        if depth != 0:
            raise Exception("Mismatched parentheses in STL expression!")

        elif len(parts) % 2 != 1:
            # Not sure this is possible
            raise Exception("Even number of parts in parenthetical expression: {}".format(phi))
        return tuple(parts)


def test():
    stl1 = 'G[0,20]F[0,2]((x>2)&(x<3)&(y>1)&(y<3))'
    stl2 = 'G[0,20]((F[0,2](x<5)) & (F[0,3](y>1)))'
    stl3 = 'G[0,20]F[0,2](((x>2)&(x<3))|((y>1)&(y<3)))'
    parser1 = STL(stl1)
    parser2 = STL(stl2)
    parser3 = STL(stl3)
    print('')
    print(stl1)
    print(parser1.get_parsed())
    print('')
    print(stl2)
    print(parser2.get_parsed())
    print('')
    print(stl3)
    print(parser3.get_parsed())



if __name__ == '__main__':
    test()