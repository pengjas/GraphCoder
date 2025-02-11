module div_32bit(
    input wire [31:0] A,   
    input wire [15:0] B,   
    output wire [31:0] result, 
    output wire [31:0] odd      
);


prep_dividend prep_dividend_inst (
    .A(A),
    .B(B),
    .prep_A(prep_A),
    .prep_B(prep_B)
);


div_operation div_operation_inst (
    .prep_A(prep_A),
    .prep_B(prep_B),
    .result(result),
    .odd(odd)
);

endmodule prep_dividend(
    input wire [31:0] A,
    input wire [15:0] B,
    output wire [31:0] prep_A,
    output wire [15:0] prep_B
);


assign prep_A = {A[31], A};
assign prep_B = {1'b0, B};

endmodule

module div_operation(
    input wire [31:0] prep_A,
    input wire [15:0] prep_B,
    output wire [31:0] result,
    output wire [31:0] odd
);


integer i;
reg [31:0] quotient;
reg [31:0] remainder;

always @(*) begin
    quotient = 0;
    remainder = prep_A;
    for (i = 0; i < 16; i = i + 1) begin
        if (remainder >= prep_B) begin
            remainder = remainder - prep_B;
            quotient = quotient + 1;
        end
        remainder = {remainder[30:0], 1'b0};
    end
    result = quotient;
    odd = remainder;
end

endmodule