(define (problem BW-99-1-3)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63 b64 b65 b66 b67 b68 b69 b70 b71 b72 b73 b74 b75 b76 b77 b78 b79 b80 b81 b82 b83 b84 b85 b86 b87 b88 b89 b90 b91 b92 b93 b94 b95 b96 b97 b98 b99 - block)
    (:init
        (handempty)
        (on-table b1)
        (on b2 b65)
        (on b3 b7)
        (on b4 b61)
        (on b5 b21)
        (on b6 b24)
        (on b7 b67)
        (on-table b8)
        (on b9 b98)
        (on b10 b96)
        (on b11 b45)
        (on b12 b8)
        (on b13 b89)
        (on b14 b37)
        (on b15 b88)
        (on b16 b13)
        (on b17 b26)
        (on-table b18)
        (on b19 b10)
        (on-table b20)
        (on b21 b32)
        (on b22 b53)
        (on-table b23)
        (on b24 b84)
        (on b25 b3)
        (on b26 b12)
        (on b27 b30)
        (on-table b28)
        (on b29 b47)
        (on b30 b11)
        (on b31 b22)
        (on b32 b49)
        (on b33 b27)
        (on b34 b91)
        (on b35 b86)
        (on b36 b5)
        (on b37 b64)
        (on b38 b87)
        (on b39 b80)
        (on b40 b18)
        (on b41 b25)
        (on b42 b28)
        (on b43 b19)
        (on b44 b38)
        (on b45 b95)
        (on b46 b36)
        (on b47 b15)
        (on b48 b23)
        (on b49 b41)
        (on b50 b58)
        (on b51 b68)
        (on b52 b63)
        (on b53 b75)
        (on b54 b1)
        (on b55 b40)
        (on b56 b85)
        (on b57 b46)
        (on b58 b44)
        (on b59 b48)
        (on b60 b70)
        (on b61 b16)
        (on b62 b6)
        (on-table b63)
        (on b64 b93)
        (on b65 b55)
        (on b66 b35)
        (on-table b67)
        (on b68 b60)
        (on b69 b73)
        (on b70 b34)
        (on-table b71)
        (on b72 b76)
        (on b73 b39)
        (on b74 b51)
        (on b75 b71)
        (on-table b76)
        (on b77 b81)
        (on b78 b29)
        (on b79 b50)
        (on b80 b31)
        (on b81 b42)
        (on b82 b17)
        (on b83 b43)
        (on-table b84)
        (on b85 b94)
        (on b86 b90)
        (on b87 b52)
        (on b88 b66)
        (on b89 b56)
        (on b90 b74)
        (on b91 b9)
        (on-table b92)
        (on b93 b99)
        (on b94 b33)
        (on b95 b20)
        (on b96 b97)
        (on b97 b4)
        (on b98 b57)
        (on b99 b82)
        (clear b2)
        (clear b14)
        (clear b54)
        (clear b59)
        (clear b62)
        (clear b69)
        (clear b72)
        (clear b77)
        (clear b78)
        (clear b79)
        (clear b83)
        (clear b92)
    )
    (:goal
        (and
            (on b1 b52)
            (on b2 b5)
            (on b3 b89)
            (on b4 b3)
            (on-table b5)
            (on b6 b69)
            (on b7 b79)
            (on b8 b68)
            (on b9 b21)
            (on b10 b66)
            (on b11 b14)
            (on b12 b85)
            (on b13 b67)
            (on b14 b34)
            (on b15 b63)
            (on b16 b81)
            (on b17 b27)
            (on b18 b77)
            (on b19 b56)
            (on b20 b86)
            (on b21 b93)
            (on-table b22)
            (on b23 b99)
            (on b24 b1)
            (on-table b25)
            (on b26 b64)
            (on b27 b72)
            (on-table b28)
            (on b29 b41)
            (on b30 b88)
            (on b31 b26)
            (on b32 b58)
            (on b33 b95)
            (on b34 b39)
            (on b35 b31)
            (on b36 b48)
            (on b37 b91)
            (on b38 b42)
            (on b39 b15)
            (on b40 b6)
            (on b41 b83)
            (on b42 b75)
            (on b43 b11)
            (on b44 b16)
            (on b45 b80)
            (on b46 b20)
            (on b47 b43)
            (on b48 b97)
            (on b49 b53)
            (on b50 b78)
            (on b51 b49)
            (on b52 b19)
            (on b53 b28)
            (on b54 b62)
            (on b55 b59)
            (on-table b56)
            (on b57 b65)
            (on-table b58)
            (on b59 b90)
            (on b60 b37)
            (on b61 b25)
            (on b62 b71)
            (on b63 b10)
            (on b64 b24)
            (on b65 b2)
            (on b66 b82)
            (on-table b67)
            (on b68 b36)
            (on b69 b29)
            (on b70 b40)
            (on b71 b44)
            (on b72 b87)
            (on b73 b47)
            (on b74 b30)
            (on b75 b55)
            (on b76 b9)
            (on-table b77)
            (on b78 b17)
            (on b79 b51)
            (on b80 b4)
            (on b81 b84)
            (on b82 b12)
            (on b83 b54)
            (on b84 b96)
            (on b85 b92)
            (on b86 b74)
            (on b87 b73)
            (on b88 b23)
            (on b89 b35)
            (on b90 b60)
            (on b91 b45)
            (on b92 b8)
            (on b93 b18)
            (on b94 b7)
            (on b95 b22)
            (on b96 b13)
            (on b97 b46)
            (on b98 b70)
            (on-table b99)
        )
    )
)