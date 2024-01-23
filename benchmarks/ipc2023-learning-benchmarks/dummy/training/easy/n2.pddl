(define (problem n-2)
(:domain dummy)
(:objects n1 r1 n2)
(:init
(at n1)
(connected n1 n2)
(connected n1 r1)
(connected r1 n2)
)
(:goal
(at n2)
)
)
