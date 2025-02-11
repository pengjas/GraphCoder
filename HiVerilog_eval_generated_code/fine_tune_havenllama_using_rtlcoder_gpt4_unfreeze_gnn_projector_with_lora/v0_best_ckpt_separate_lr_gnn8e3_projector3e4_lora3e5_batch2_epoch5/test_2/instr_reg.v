module instr_reg (
    input clk,
    input rst,
    input [1:0] fetch,
    input [7:0] data,
    output [2:0] ins,
    output [4:0] ad1,
    output [7:0] ad2
);
reg [7:0] ins_p1;
reg [7:0] ins_p2;

always @(posedge clk) begin
    if (rst == 1'b0) begin
        ins_p1 <= 8'b0;
        ins_p2 <= 8'b0;
    end
    else begin
        if (fetch == 2'b01) begin
            ins_p1 <= data;
        end
        else if (fetch == 2'b10) begin
            ins_p2 <= data;
        end
    end
end

assign ins = {ins_p1[2:0], ins_p2[2:0]};
assign ad1 = {ins_p1[4:0], ins_p2[4:0]};
assign ad2 = {ins_p1[7:0], ins_p2[7:0]};

endmodule