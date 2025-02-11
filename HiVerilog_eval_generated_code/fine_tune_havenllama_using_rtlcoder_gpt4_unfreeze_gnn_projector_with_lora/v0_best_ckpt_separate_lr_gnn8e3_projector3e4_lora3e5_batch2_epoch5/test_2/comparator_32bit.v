module subtraction_32bit(
    input [31:0] A,   
    input [31:0] B,   
    output [31:0] C    
);


module carry_generation_32bit(
    input [31:0] A,   
    input [31:0] B,   
    input [31:0] C,   
    output [31:0] D    
);


module subtraction_32bit(
    input [31:0] A,   
    input [31:0] B,   
    output [31:0] C    
);


module carry_generation_32bit(
    input [31:0] A,   
    input [31:0] B,   
    input [31:0] C,   
    output [31:0] D    
);


module comparator_32bit(
    input [31:0] A,   
    input [31:0] B,   
    output A_greater, 
    output A_equal,   
    output A_less     
);


subtraction_32bit sub_inst(
    .A(A[31:0]),
    .B(B[31:0]),
    .C(C[31:0])
);


carry_generation_32bit carry_inst(
    .A(A[31:0]),
    .B(B[31:0]),
    .C(C[31:0]),
    .D(D[31:0])
);


comparator_32bit comp_inst(
    .A(A[31:0]),
    .B(B[31:0]),
    .A_greater(A_greater),
    .A_equal(A_equal),
    .A_less(A_less)
);

endmodule