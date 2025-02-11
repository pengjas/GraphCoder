module freq_div (
    input CLK_in,
    input RST,
    output CLK_50,
    output CLK_10,
    output CLK_1
);

reg [31:0] counter_50;
reg [31:0] counter_10;
reg [31:0] counter_1;

reg CLK_50, CLK_10, CLK_1;


always @(CLK_in or RST) begin
    if (RST) begin
        counter_50 <= 0;
        counter_10 <= 0;
        counter_1 <= 0;
        CLK_50 <= 0;
        CLK_10 <= 0;
        CLK_1 <= 0;
    end else begin
        if (CLK_in) begin
            if (counter_50 == 3'b111) begin
                counter_50 <= 0;
                CLK_50 <= ~CLK_50;
            end else begin
                counter_50 <= counter_50 + 1;
                CLK_50 <= CLK_50;
            end

            if (counter_10 == 3'b111) begin
                counter_10 <= 0;
                CLK_10 <= ~CLK_10;
            end else begin
                counter_10 <= counter_10 + 1;
                CLK_10 <= CLK_10;
            end

            if (counter_1 == 3'b111) begin
                counter_1 <= 0;
                CLK_1 <= ~CLK_1;
            end else begin
                counter_1 <= counter_1 + 1;
                CLK_1 <= CLK_1;
            end
        end
    end
end

endmodule

module clk_div_2 (
    input CLK_in,
    input RST,
    output CLK_out
);

reg CLK_out_reg;

always @(posedge CLK_in or posedge RST) begin
    if (RST) begin
        CLK_out_reg <= 0;
    end else begin
        CLK_out_reg <= ~CLK_out_reg;
    end
end

assign CLK_out = CLK_out_reg;

endmodule

module clk_div_10 (
    input CLK_in,
    input RST,
    output CLK_out
);

reg [3:0] counter;
reg CLK_out_reg;

always @(posedge CLK_in or posedge RST) begin
    if (RST) begin
        counter <= 0;
        CLK_out_reg <= 0;
    end else begin
        if (counter == 4) begin
            counter <= 0;
            CLK_out_reg <= ~CLK_out_reg;
        end else begin
            counter <= counter + 1;
            CLK_out_reg <= CLK_out_reg;
        end
    end
end

assign CLK_out = CLK_out_reg;

endmodule


module clk_div_100 (
    input CLK_in,
    input RST,
    output CLK_out
);

reg [5:0] counter;
reg CLK_out_reg;

always @(posedge CLK_in or posedge RST) begin
    if (RST) begin
        counter <= 0;
        CLK_out_reg <= 0;
    end else begin
        if (counter == 6'b111111) begin
            counter <= 0;
            CLK_out_reg <= ~CLK_out_reg;
        end else begin
            counter <= counter + 1;
            CLK_out_reg <= CLK_out_reg;
        end
    end
end

assign CLK_out = CLK_out_reg;

endmodule

endmodule