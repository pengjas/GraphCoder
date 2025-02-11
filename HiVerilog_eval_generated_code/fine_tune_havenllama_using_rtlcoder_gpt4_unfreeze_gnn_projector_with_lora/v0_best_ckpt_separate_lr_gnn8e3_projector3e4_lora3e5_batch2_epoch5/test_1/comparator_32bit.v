module subtraction_32bit(
    input [31:0] A,   
    input [31:0] B,   
    output [31:0] carry, 
    output [31:0] result  
);


module comparator_32bit(
    input [31:0] A,   
    input [31:0] B,   
    output A_greater, 
    output A_equal,   
    output A_less     
);


subtraction_32bit sub_inst(
    .A(A),
    .B(B),
    .carry(carry),
    .result(result)
);


assign A_greater = (result != 0);
assign A_equal = (carry == 0);
assign A_less = (carry == 1);

endmodule


module subtraction_32bit(
    input [31:0] A,   
    input [31:0] B,   
    output [31:0] carry, 
    output [31:0] result  
);


assign result = A - B;
assign carry = (result[31] == 1) ? 1 : 0;

endmodule


module comparator_32bit(
    input [31:0] A,   
    input [31:0] B,   
    output A_greater, 
    output A_equal,   
    output A_less     
);


subtraction_32bit sub_inst(
    .A(A),
    .B(B),
    .carry(carry),
    .result(result)
);


assign A_greater = (result != 0);
assign A_equal = (carry == 0);
assign A_less = (carry == 1);

endmodule