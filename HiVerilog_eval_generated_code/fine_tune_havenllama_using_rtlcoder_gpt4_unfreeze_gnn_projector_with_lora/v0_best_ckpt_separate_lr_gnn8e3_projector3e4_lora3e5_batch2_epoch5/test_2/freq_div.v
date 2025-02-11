module freq_div (
    input CLK_in,
    input RST,
    output CLK_50,
    output CLK_10,
    output CLK_1
);

    reg CLK_50_reg, CLK_10_reg, CLK_1_reg;
    reg [3:0] count_50, count_10, count_1000;


    always @(posedge(CLK_in) begin
        if (RST) begin
            CLK_50_reg <= 0;
            CLK_10_reg <= 0;
            CLK_1_reg <= 0;
            count_50 <= 0;
            count_10 <= 0;
            count_100 <= 0;
        end
        else begin
            CLK_50_reg <= CLK_50_reg ^ CLK_in;
            CLK_10_reg <= CLK_10_reg ^ CLK_in;
            CLK_1_reg <= CLK_1_reg ^ CLK_in;
            if (count_50 == 1) begin
                count_50 <= 0;
                CLK_50_reg <= ~CLK_50_reg;
            end
            else begin
                count_50 <= count_50 + 1;
            end
            if (count_10 == 1) begin
                count_10 <= 0;
                CLK_10_reg <= ~CLK_10_reg;
            end
            else begin
                count_10 <= count_10 + 1;
            end
            if (count_100 == 1) begin
                count_100 <= 0;
                CLK_1_reg <= ~CLK_1_reg;
            end
            else begin
                count_100 <= count_100 + 1;
            end
        end
    end

    assign CLK_50 = CLK_50_reg;
    assign CLK_10 = CLK_10_reg;
    assign CLK_1 = CLK_1_reg;

endmodule

module clk_div_2 (
    input CLK_in,
    input RST,
    output CLK_out
);
    reg CLK_out_reg;
    reg [1:0] count;

    always @(posedge CLK_in) begin
        if (RST) begin
            CLK_out_reg <= 0;
            count <= 0;
        end
        else begin
            if (count == 2) begin
                count <= 0;
                CLK_out_reg <= ~CLK_out_reg;
            end
            else begin
                count <= count + 1;
            end
        end
    end

    assign CLK_out = CLK_out_reg;

endmodule


module clk_div_10 (
    input CLK_in,
    input RST,
    output CLK_out
);
    reg CLK_out_reg;
    reg [3:0] count;

    always @(posedge CLK_in) begin
        if (RST) begin
            CLK_out_reg <= 0;
            count <= 0;
        end
        else begin
            if (count == 4) begin
                count <= 0;
                CLK_out_reg <= ~CLK_out_reg;
            end
            else begin
                count <= count + 1;
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
    reg CLK_out_reg;
    reg [4:0] count;

    always @(posedge CLK_in) begin
        if (RST) begin
            CLK_out_reg <= 0;
            count <= 0;
        end
        else begin
            if (count == 49) begin
                count <= 0;
                CLK_out_reg <= ~CLK
endmodule