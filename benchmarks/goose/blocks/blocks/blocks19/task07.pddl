(define (problem BW-19-1-7)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 - block)
    (:init
        (handempty)
        (on b1 b16)
        (on-table b2)
        (on b3 b13)
        (on b4 b7)
        (on-table b5)
        (on b6 b5)
        (on b7 b19)
        (on b8 b17)
        (on-table b9)
        (on-table b10)
        (on b11 b18)
        (on-table b12)
        (on b13 b2)
        (on b14 b10)
        (on b15 b4)
        (on b16 b3)
        (on b17 b6)
        (on b18 b8)
        (on b19 b14)
        (clear b1)
        (clear b9)
        (clear b11)
        (clear b12)
        (clear b15)
    )
    (:goal
        (and
            (on b1 b10)
            (on b2 b14)
            (on b3 b8)
            (on-table b4)
            (on b5 b17)
            (on-table b6)
            (on b7 b18)
            (on-table b8)
            (on b9 b13)
            (on b10 b7)
            (on b11 b19)
            (on b12 b3)
            (on b13 b1)
            (on b14 b4)
            (on b15 b2)
            (on b16 b6)
            (on b17 b15)
            (on b18 b16)
            (on b19 b12)
        )
    )
)