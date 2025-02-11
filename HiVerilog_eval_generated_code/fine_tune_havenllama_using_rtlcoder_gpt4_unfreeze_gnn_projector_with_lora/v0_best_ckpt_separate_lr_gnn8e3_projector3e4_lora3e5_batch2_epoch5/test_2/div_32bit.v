module div_32bit(
    input wire [31:0] A,   
    input wire [15:0] B,   
    output wire [31:0] result, 
    output wire [31:0] odd      
);


prep_dividend prep_dividend_inst (
    .A(A),
    .B(B),
    .result(result),
    .odd(odd)
);


div_operation div_op_inst (
    .A(A),
    .B(B),
    .result(result),
    .odd(odd)
);

endmodule


module prep_dividend(
    input wire [315:0] A,   
    input wire [15:0] B,   
    output wire [31:0] result, 
    output wire [31:0] odd      
);


assign result = {1'b0, A[15:0]};
assign odd = {1'b0, A[31:16]};

endmodule


module div_operation(
    input wire [31:0] A,   
    input wire [15:0] B,   
    output wire [31:0] result, 
    output wire [31:0] odd      
);


wire [15:0] shifted_dividend;
wire [15:0] shifted_divisor;
wire [31:0] shifted_result;
wire [31:0] shifted_odd;

assign shifted_dividend = {A[31:16], 16'b0};
assign shifted_divisor = {B, 16'b0};
assign shifted_result = {32'b0, 1'b0};
assign shifted_odd = {32'b0, 1'b0};

genvar i;
generate
    for (i = 0; i < 16; i = i + 1) begin
        if (shifted_dividend[15:0] >= shifted_divisor) begin
            shifted_result[31:16] = shifted_result[31:16] + 1;
            shifted_dividend[31:16] = shifted_dividend[31:16] - shifted_divisor;
        end
        shifted_divisor = shifted_divisor << 1;
    end
endgenerate


assign result = shifted_result[31:16];
assign odd = shifted_odd[31:16];

endmodule