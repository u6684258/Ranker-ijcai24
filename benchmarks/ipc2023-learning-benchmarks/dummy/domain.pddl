
(define (domain dummy)
  (:requirements :strips)
  (:predicates (connected ?x ?y)
	       (at ?x)
  )

  (:action go
	     :parameters (?x ?y)
	     :precondition (and (at ?x) (connected ?x ?y))
	     :effect
	     (and (not (at ?x)) (at ?y) (not (connected ?x ?y)))

  )

)
