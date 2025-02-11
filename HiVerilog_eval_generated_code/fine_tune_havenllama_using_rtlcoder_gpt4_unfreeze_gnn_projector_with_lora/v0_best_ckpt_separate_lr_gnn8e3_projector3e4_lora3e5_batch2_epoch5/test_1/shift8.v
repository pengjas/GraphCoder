module shift8 (
    input clk,
    input [7:0] d,
    input [1:0] sel,
    output reg [7:0] q
);

    
    reg [7:0] q1, q2, q3;

    
    reg [1:0] sel_reg;

    
    reg [7:0] shift_reg [2:0];

    
    always @(posedge clk) begin
        q1 <= d;
        q2 <= q1;
        q3 <= q2;
    end

    
    always @(posedge clk) begin
        shift_reg[0] <= q1;
        shift_reg[1] <= shift_reg[0];
        shift_reg[2] <= shift_reg[1];
    end

    
    always @(posedge clk) begin
        sel_reg <= sel;
    end

    
    always @(posedge clk) begin
        case (sel_reg)
            2'b00: q <= shift_reg[0];
            2'b01: q <= shift_reg[1];
            2'b10: q <= shift_reg[2];
            2'b11: q <= d;
            default: q <= 8'b0;
        endcase
    end
endmodule


module my_dff8 (
    input clk,
    input [7:0] d,
    output reg [7:0] q
);

    always @(posedge clk) begin
        q <= d;
    end

endmodule


module my_mux4to1 (
    input [1:0] sel,
    input [3:0] d,
    output reg [3:0] q
);

    always @(*) begin
        case (sel)
            2'b00: q <= d[0];
            2'b01: q <= d[1];
            2'b10: q <= d[2];
            2'b11: q <= d[3];
            default: q <= 4'b0;
        endcase
    end

endmodule