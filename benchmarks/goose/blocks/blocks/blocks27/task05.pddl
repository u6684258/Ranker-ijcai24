(define (problem BW-27-1-5)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b20)
        (on b3 b6)
        (on b4 b25)
        (on b5 b9)
        (on-table b6)
        (on b7 b2)
        (on b8 b7)
        (on b9 b23)
        (on b10 b21)
        (on b11 b16)
        (on b12 b27)
        (on-table b13)
        (on-table b14)
        (on b15 b12)
        (on b16 b3)
        (on b17 b4)
        (on b18 b14)
        (on b19 b5)
        (on b20 b24)
        (on b21 b11)
        (on-table b22)
        (on b23 b18)
        (on b24 b10)
        (on b25 b22)
        (on b26 b13)
        (on b27 b8)
        (clear b1)
        (clear b15)
        (clear b17)
        (clear b19)
        (clear b26)
    )
    (:goal
        (and
            (on b1 b6)
            (on-table b2)
            (on-table b3)
            (on b4 b24)
            (on b5 b21)
            (on-table b6)
            (on b7 b12)
            (on-table b8)
            (on b9 b13)
            (on b10 b15)
            (on b11 b1)
            (on b12 b5)
            (on b13 b20)
            (on b14 b9)
            (on b15 b27)
            (on b16 b22)
            (on-table b17)
            (on b18 b7)
            (on b19 b18)
            (on b20 b26)
            (on b21 b14)
            (on b22 b2)
            (on-table b23)
            (on b24 b8)
            (on-table b25)
            (on-table b26)
            (on b27 b4)
        )
    )
)