module alu_add(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a + b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_sub(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a - b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_and(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a & b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_or(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a | b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_xor(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a ^ b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_slt(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a < b ? a : b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_sltu(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a < b ? b : a;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_sll(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a << b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_srl(
    input [31:0] a,
    input [31:0] b,
    output [31:0] r,
    output zero,
    output carry
);

reg [31:0] temp;

always @*
begin
    temp = a >> b;
    r = temp[31:0];
    zero = (temp == 0);
    carry = (temp[32]);
end

module alu_sra(
endmodule