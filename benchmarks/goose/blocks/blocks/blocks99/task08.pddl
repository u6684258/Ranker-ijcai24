(define (problem BW-99-1-8)
    (:domain blocksworld)
    (:objects b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 b27 b28 b29 b30 b31 b32 b33 b34 b35 b36 b37 b38 b39 b40 b41 b42 b43 b44 b45 b46 b47 b48 b49 b50 b51 b52 b53 b54 b55 b56 b57 b58 b59 b60 b61 b62 b63 b64 b65 b66 b67 b68 b69 b70 b71 b72 b73 b74 b75 b76 b77 b78 b79 b80 b81 b82 b83 b84 b85 b86 b87 b88 b89 b90 b91 b92 b93 b94 b95 b96 b97 b98 b99 - block)
    (:init
        (handempty)
        (on b1 b56)
        (on b2 b87)
        (on-table b3)
        (on b4 b71)
        (on b5 b16)
        (on b6 b54)
        (on b7 b12)
        (on b8 b84)
        (on b9 b50)
        (on b10 b38)
        (on b11 b10)
        (on-table b12)
        (on b13 b6)
        (on b14 b95)
        (on b15 b22)
        (on b16 b44)
        (on-table b17)
        (on b18 b97)
        (on b19 b90)
        (on b20 b59)
        (on b21 b73)
        (on-table b22)
        (on b23 b79)
        (on b24 b19)
        (on b25 b76)
        (on b26 b7)
        (on b27 b41)
        (on b28 b57)
        (on b29 b99)
        (on b30 b46)
        (on b31 b26)
        (on b32 b88)
        (on b33 b13)
        (on b34 b78)
        (on b35 b37)
        (on b36 b49)
        (on b37 b23)
        (on b38 b17)
        (on b39 b62)
        (on b40 b69)
        (on b41 b61)
        (on b42 b96)
        (on b43 b89)
        (on b44 b9)
        (on b45 b18)
        (on b46 b31)
        (on b47 b29)
        (on b48 b98)
        (on b49 b42)
        (on b50 b32)
        (on b51 b25)
        (on b52 b55)
        (on b53 b93)
        (on b54 b14)
        (on b55 b48)
        (on b56 b75)
        (on b57 b1)
        (on-table b58)
        (on b59 b91)
        (on b60 b27)
        (on b61 b40)
        (on b62 b70)
        (on b63 b47)
        (on b64 b77)
        (on b65 b43)
        (on b66 b82)
        (on b67 b51)
        (on b68 b11)
        (on b69 b94)
        (on b70 b86)
        (on-table b71)
        (on b72 b92)
        (on-table b73)
        (on-table b74)
        (on b75 b30)
        (on b76 b21)
        (on b77 b58)
        (on b78 b67)
        (on b79 b39)
        (on b80 b3)
        (on b81 b63)
        (on b82 b33)
        (on b83 b4)
        (on b84 b65)
        (on b85 b24)
        (on-table b86)
        (on b87 b5)
        (on b88 b20)
        (on b89 b85)
        (on b90 b53)
        (on b91 b83)
        (on b92 b64)
        (on b93 b34)
        (on b94 b45)
        (on b95 b36)
        (on b96 b72)
        (on b97 b66)
        (on b98 b80)
        (on b99 b28)
        (clear b2)
        (clear b8)
        (clear b15)
        (clear b35)
        (clear b52)
        (clear b60)
        (clear b68)
        (clear b74)
        (clear b81)
    )
    (:goal
        (and
            (on b1 b11)
            (on b2 b65)
            (on-table b3)
            (on b4 b42)
            (on b5 b96)
            (on b6 b38)
            (on b7 b76)
            (on b8 b26)
            (on b9 b34)
            (on b10 b30)
            (on b11 b50)
            (on b12 b46)
            (on b13 b77)
            (on b14 b15)
            (on b15 b67)
            (on b16 b41)
            (on b17 b25)
            (on b18 b24)
            (on b19 b32)
            (on b20 b29)
            (on b21 b78)
            (on b22 b16)
            (on b23 b53)
            (on b24 b1)
            (on-table b25)
            (on-table b26)
            (on b27 b64)
            (on b28 b61)
            (on b29 b22)
            (on b30 b88)
            (on b31 b54)
            (on b32 b2)
            (on b33 b89)
            (on b34 b72)
            (on b35 b52)
            (on b36 b39)
            (on b37 b94)
            (on b38 b92)
            (on b39 b44)
            (on b40 b70)
            (on b41 b19)
            (on b42 b7)
            (on b43 b8)
            (on b44 b84)
            (on b45 b20)
            (on b46 b63)
            (on b47 b90)
            (on b48 b23)
            (on b49 b28)
            (on b50 b83)
            (on b51 b21)
            (on b52 b99)
            (on b53 b93)
            (on b54 b14)
            (on b55 b49)
            (on b56 b91)
            (on b57 b36)
            (on b58 b59)
            (on b59 b55)
            (on b60 b66)
            (on b61 b56)
            (on b62 b86)
            (on b63 b3)
            (on b64 b47)
            (on b65 b69)
            (on b66 b68)
            (on b67 b10)
            (on-table b68)
            (on b69 b35)
            (on b70 b82)
            (on b71 b27)
            (on b72 b45)
            (on b73 b43)
            (on b74 b62)
            (on b75 b9)
            (on b76 b37)
            (on b77 b57)
            (on b78 b87)
            (on b79 b73)
            (on b80 b51)
            (on b81 b33)
            (on b82 b85)
            (on-table b83)
            (on-table b84)
            (on b85 b17)
            (on b86 b97)
            (on b87 b74)
            (on b88 b6)
            (on b89 b31)
            (on b90 b95)
            (on b91 b79)
            (on b92 b60)
            (on b93 b81)
            (on b94 b40)
            (on b95 b4)
            (on b96 b58)
            (on b97 b98)
            (on b98 b75)
            (on b99 b18)
        )
    )
)