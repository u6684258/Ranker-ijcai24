(define (problem BW-24-1-10)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 - block)
    (:init
        (handempty)
        (on b1 b3)
        (on b2 b9)
        (on b3 b10)
        (on b4 b12)
        (on-table b5)
        (on b6 b5)
        (on b7 b19)
        (on-table b8)
        (on b9 b1)
        (on b10 b15)
        (on b11 b16)
        (on b12 b18)
        (on b13 b11)
        (on b14 b22)
        (on b15 b4)
        (on b16 b23)
        (on b17 b7)
        (on-table b18)
        (on b19 b6)
        (on-table b20)
        (on b21 b14)
        (on b22 b13)
        (on b23 b20)
        (on-table b24)
        (clear b2)
        (clear b8)
        (clear b17)
        (clear b21)
        (clear b24)
    )
    (:goal
        (and
            (on b1 b14)
            (on b2 b6)
            (on b3 b22)
            (on b4 b2)
            (on-table b5)
            (on b6 b11)
            (on-table b7)
            (on-table b8)
            (on b9 b23)
            (on-table b10)
            (on b11 b17)
            (on b12 b1)
            (on-table b13)
            (on-table b14)
            (on b15 b10)
            (on b16 b9)
            (on b17 b8)
            (on b18 b21)
            (on b19 b5)
            (on b20 b13)
            (on b21 b15)
            (on b22 b4)
            (on b23 b19)
            (on b24 b12)
        )
    )
)